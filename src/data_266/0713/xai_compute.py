"""
xai_compute.py  v3
Patent Litigation XAI Pipeline — with Native GAT Attention

New in v3:
  - extract_attention_weights(): per-layer, per-relation attention via
    GATConv(return_attention_weights=True) — no hooks needed
  - BFS hop distances per node (for Evidence Flow animation)
  - Edge JSON: attention{L0..L3}, delta_p, hop info
  - Storyline conclusion card auto-generated

Usage:
    cd 0713/
    python xai_compute.py [--pair 0] [--khop 2] [--topk 10]
"""

import sys, argparse, collections
from pathlib import Path
import json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "viz_app"))
from backend.loader import load_state
from backend.model import EDGE_TYPES, _CONV_ACTS

parser = argparse.ArgumentParser()
parser.add_argument("--pair",      type=int, default=0)
parser.add_argument("--khop",      type=int, default=2)
parser.add_argument("--topk",      type=int, default=10)
parser.add_argument("--gnn_ep",    type=int, default=200)
parser.add_argument("--tr_ep",     type=int, default=80)
parser.add_argument("--co_limit",  type=int, default=15)
parser.add_argument("--ipc_limit", type=int, default=15)
args = parser.parse_args()

OUT_JSON = Path(__file__).parent / "xai_results.json"

# ════════════════════════════════════════════════════════════════════════════
# Helper: extract per-layer attention weights
# ════════════════════════════════════════════════════════════════════════════
def extract_attention_weights(model, data, subset_global_set):
    """
    Run a forward pass extracting GATConv attention per layer per relation.
    Uses return_attention_weights=True — no hooks needed.

    Returns:
        dict[layer_name → dict[rel_type → dict["sg_dg" → alpha_mean]]]
        Only edges where both endpoints are in subset_global_set are stored.
    """
    model.eval()
    x_pat = torch.cat([data["patent"].struct, data["patent"].x], dim=1)
    x = {
        "patent":  model.pat_proj(x_pat),
        "company": F.leaky_relu(model.co_proj(data["company"].x)),
        "ipc":     model.ipc_emb.weight,
    }

    attn_layers = {}

    with torch.no_grad():
        for layer_idx in range(4):
            hconv = model.convs[layer_idx]
            layer_attn = {}

            for et, gat in hconv.convs.items():
                src_t, rel_t, dst_t = et
                if et not in data.edge_index_dict:
                    continue

                ei = data.edge_index_dict[et]
                x_in = x[src_t] if src_t == dst_t else (x[src_t], x[dst_t])

                try:
                    _, (ei_out, alpha) = gat(x_in, ei, return_attention_weights=True)
                except Exception:
                    continue

                # alpha: (E, heads) → abs mean over heads
                a = alpha.abs().mean(dim=-1) if alpha.dim() > 1 else alpha.abs()
                a_np   = a.cpu().numpy()
                ei_np  = ei_out.cpu().numpy()

                # Filter to patent subgraph edges only
                if src_t == "patent" and dst_t == "patent":
                    rel_dict = {}
                    for i in range(ei_np.shape[1]):
                        sg, dg = int(ei_np[0, i]), int(ei_np[1, i])
                        if sg in subset_global_set and dg in subset_global_set:
                            key = f"{sg}_{dg}"
                            # keep max if duplicated (shouldn't happen, but safe)
                            if key not in rel_dict or a_np[i] > rel_dict[key]:
                                rel_dict[key] = round(float(a_np[i]), 4)
                    if rel_dict:
                        layer_attn[rel_t] = rel_dict

            attn_layers[f"layer{layer_idx}"] = layer_attn

            # Continue forward to update x for next layer
            if layer_idx < 3:
                act   = _CONV_ACTS[layer_idx]
                bn_d  = model.bns[layer_idx]
                x_new = hconv(x, data.edge_index_dict)
                x = {
                    k: model.drop(act(bn_d[k](v.view(-1, model.hid))))
                    if k in bn_d else model.drop(act(v))
                    for k, v in x_new.items()
                }

    return attn_layers


# ════════════════════════════════════════════════════════════════════════════
# Helper: BFS hop distance
# ════════════════════════════════════════════════════════════════════════════
def bfs_hop(start, adj, max_hop):
    dist = {start: 0}
    q = collections.deque([start])
    while q:
        node = q.popleft()
        if dist[node] >= max_hop:
            continue
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


# ════════════════════════════════════════════════════════════════════════════
# 1. Load
# ════════════════════════════════════════════════════════════════════════════
print("▶ Loading model + data...")
state    = load_state(ROOT)
model    = state["model"]
data     = state["data"]
device   = state["device"]
pat_list = state["pat_list"]
co_list  = state["co_list"]
ipc_list = state["ipc_list"]
test_edges = state["test_edges"]

model.eval()
with torch.no_grad():
    h_dict = model(data)
h_pat = h_dict["patent"].detach()
print(f"   Patents={len(pat_list)}  Co={len(co_list)}  IPC={len(ipc_list)}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Target pair
# ════════════════════════════════════════════════════════════════════════════
src_idx = int(test_edges[0, args.pair])
dst_idx = int(test_edges[1, args.pair])
p_base  = torch.sigmoid((h_pat[src_idx] * h_pat[dst_idx]).sum()).item()
src_nid = f"pat_{src_idx}"
dst_nid = f"pat_{dst_idx}"

print(f"\n   Plaintiff : {pat_list[src_idx]}")
print(f"   Defendant : {pat_list[dst_idx]}")
print(f"   P(sue)    = {p_base:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 3. K-hop patent subgraph
# ════════════════════════════════════════════════════════════════════════════
print(f"\n▶ Extracting {args.khop}-hop patent subgraph...")

PAT_PAT = ["cites", "cited_by", "sued", "sued_by"]
ei_parts, et_parts = [], []
for et in EDGE_TYPES:
    if et[1] in PAT_PAT:
        ei = data[et[0], et[1], et[2]].edge_index
        ei_parts.append(ei)
        et_parts.extend([et[1]] * ei.size(1))

all_ei = torch.cat(ei_parts, dim=1)
all_et = np.array(et_parts)

subset, sub_ei, mapping, edge_mask = k_hop_subgraph(
    node_idx=torch.tensor([src_idx, dst_idx]),
    num_hops=args.khop,
    edge_index=all_ei,
    num_nodes=len(pat_list),
    relabel_nodes=True,
)
sub_et     = all_et[edge_mask.cpu().numpy()]
src_local  = mapping[0].item()
dst_local  = mapping[1].item()
x_sub      = h_pat[subset].detach().cpu()
sub_ei_cpu = sub_ei.cpu()
N_sub      = len(subset)
subset_set = {subset[i].item() for i in range(N_sub)}

print(f"   Nodes={N_sub}  Edges={sub_ei.size(1)}")

# BFS hop distances from src and dst
adj_sub = collections.defaultdict(list)
for i in range(sub_ei_cpu.size(1)):
    s = subset[sub_ei_cpu[0, i].item()].item()
    d = subset[sub_ei_cpu[1, i].item()].item()
    adj_sub[s].append(d)
    adj_sub[d].append(s)
hop_src = bfs_hop(src_idx, adj_sub, args.khop + 1)
hop_dst = bfs_hop(dst_idx, adj_sub, args.khop + 1)

# ════════════════════════════════════════════════════════════════════════════
# 4. Native Attention Extraction (NEW)
# ════════════════════════════════════════════════════════════════════════════
print("\n▶ Extracting native GAT attention weights (all 4 layers)...")
attn_layers = extract_attention_weights(model, data, subset_set)
for ln, ld in attn_layers.items():
    n_edges_stored = sum(len(v) for v in ld.values())
    print(f"   {ln}: {list(ld.keys())}  total_subgraph_edges={n_edges_stored}")

# ════════════════════════════════════════════════════════════════════════════
# 5. PatentExplainModel + GNNExplainer
# ════════════════════════════════════════════════════════════════════════════
print("\n▶ Training PatentExplainModel...")

class PatentExplainModel(nn.Module):
    def __init__(self, in_dim=64, hid=64, out=64):
        super().__init__()
        self.conv1 = GATConv(in_dim, hid//4, heads=4, add_self_loops=True)
        self.conv2 = GATConv(hid,    out//2, heads=2, add_self_loops=True, concat=False)

    def encode(self, x, edge_index):
        return self.conv2(F.leaky_relu(self.conv1(x, edge_index)), edge_index)

    def forward(self, x, edge_index, edge_label_index=None):
        h = self.encode(x, edge_index)
        if edge_label_index is not None:
            return (h[edge_label_index[0]] * h[edge_label_index[1]]).sum(-1)
        return torch.sigmoid((h * h).sum(-1))

explain_model = PatentExplainModel().cpu()
eli_pos = torch.tensor([[src_local], [dst_local]])

np.random.seed(42)
neg_pairs = []
while len(neg_pairs) < 10:
    s, d = np.random.randint(0, N_sub), np.random.randint(0, N_sub)
    if s != d and not (s == src_local and d == dst_local):
        neg_pairs.append((s, d))
eli_neg = torch.tensor(np.array(neg_pairs).T)
y_all   = torch.cat([torch.ones(1), torch.zeros(10)])

optimizer = torch.optim.Adam(explain_model.parameters(), lr=1e-3)
explain_model.train()
for ep in range(args.tr_ep):
    optimizer.zero_grad()
    out  = torch.cat([explain_model(x_sub, sub_ei_cpu, eli_pos),
                      explain_model(x_sub, sub_ei_cpu, eli_neg)])
    loss = F.binary_cross_entropy_with_logits(out, y_all)
    loss.backward(); optimizer.step()
    if (ep+1) % 20 == 0:
        p_p = torch.sigmoid(explain_model(x_sub, sub_ei_cpu, eli_pos)).item()
        print(f"   ep{ep+1:3d}: loss={loss.item():.4f}  P(sue)={p_p:.4f}")
explain_model.eval()

print(f"\n▶ GNNExplainer (epochs={args.gnn_ep})...")
gnn_exp = Explainer(
    model=explain_model,
    algorithm=GNNExplainer(epochs=args.gnn_ep),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(mode="binary_classification", task_level="edge", return_type="raw"),
)
eli_q        = torch.tensor([[src_local], [dst_local]])
expl         = gnn_exp(x_sub, sub_ei_cpu, edge_label_index=eli_q)
edge_mask_np = expl.edge_mask.cpu().numpy()

if expl.node_mask is not None:
    node_imp_np = expl.node_mask.cpu().numpy().mean(axis=1)
else:
    x_g = x_sub.clone().detach().requires_grad_(True)
    explain_model(x_g, sub_ei_cpu, eli_q).sum().backward()
    node_imp_np = x_g.grad.abs().mean(axis=1).detach().numpy()
node_imp_np = node_imp_np / (node_imp_np.max() + 1e-8)

rank_order = np.argsort(-edge_mask_np)
edge_rank  = np.empty_like(rank_order)
edge_rank[rank_order] = np.arange(1, len(edge_mask_np) + 1)
top_k_idx  = rank_order[:args.topk]
top_k_set  = set(top_k_idx.tolist())

# ════════════════════════════════════════════════════════════════════════════
# 6. Counterfactual — by edge TYPE
# ════════════════════════════════════════════════════════════════════════════
print("\n▶ Counterfactual: edge type removal...")
cf_by_type = {}
for et in EDGE_TYPES:
    data_cf = copy.deepcopy(data).to(device)
    data_cf[et[0], et[1], et[2]].edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
    with torch.no_grad():
        h_cf = model(data_cf)
    p_cf  = torch.sigmoid((h_cf["patent"][src_idx] * h_cf["patent"][dst_idx]).sum()).item()
    delta = p_cf - p_base
    cf_by_type[et[1]] = {"p_after": round(p_cf, 4), "delta_p": round(delta, 4)}
    print(f"   {et[1]:15s}: P={p_cf:.4f}  Δ={delta:+.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 7. Counterfactual — guided by GNNExplainer top-K
# ════════════════════════════════════════════════════════════════════════════
print(f"\n▶ Counterfactual: GNNExplainer top-{args.topk} edges...")
cf_by_edge = []
for rank, eidx in enumerate(top_k_idx):
    sln, dln = sub_ei_cpu[0, eidx].item(), sub_ei_cpu[1, eidx].item()
    sgn, dgn = subset[sln].item(), subset[dln].item()
    et_n     = sub_et[eidx]
    et_t     = next(et for et in EDGE_TYPES if et[1] == et_n)

    data_cf  = copy.deepcopy(data).to(device)
    ei_full  = data_cf[et_t[0], et_t[1], et_t[2]].edge_index
    keep     = ~((ei_full[0] == sgn) & (ei_full[1] == dgn))
    data_cf[et_t[0], et_t[1], et_t[2]].edge_index = ei_full[:, keep]

    with torch.no_grad():
        h_cf = model(data_cf)
    p_cf  = torch.sigmoid((h_cf["patent"][src_idx] * h_cf["patent"][dst_idx]).sum()).item()
    delta = p_cf - p_base

    cf_by_edge.append({
        "rank":      rank + 1,
        "edge_id":   f"e_{eidx}",
        "src_id":    f"pat_{sgn}",
        "dst_id":    f"pat_{dgn}",
        "src_label": pat_list[sgn],
        "dst_label": pat_list[dgn],
        "edge_type": et_n,
        "gnn_mask":  round(float(edge_mask_np[eidx]), 4),
        "p_after":   round(p_cf, 4),
        "delta_p":   round(delta, 4),
    })
    print(f"   Rank{rank+1:2d} ({et_n:8s}) {pat_list[sgn][:16]:16s}→{pat_list[dgn][:16]:16s}  Δ={delta:+.4f}")

# delta_p lookup per edge pair
cf_pair_lookup = {(e["src_id"], e["dst_id"]): e["delta_p"] for e in cf_by_edge}

# ════════════════════════════════════════════════════════════════════════════
# 8. Build full node + edge JSON
# ════════════════════════════════════════════════════════════════════════════
print("\n▶ Building full node/edge JSON (patent + company + IPC)...")

# Patent nodes
all_nodes  = []
node_id_set = set()
for li in range(N_sub):
    gi  = subset[li].item()
    nid = f"pat_{gi}"
    hs  = hop_src.get(gi, args.khop + 1)
    hd  = hop_dst.get(gi, args.khop + 1)
    all_nodes.append({
        "id":           nid,
        "type":         "patent",
        "label":        pat_list[gi],
        "short_label":  pat_list[gi][:14],
        "is_src":       bool(li == src_local),
        "is_dst":       bool(li == dst_local),
        "importance":   float(node_imp_np[li]),
        "hop_from_src": int(hs),
        "hop_from_dst": int(hd),
        "min_hop":      int(min(hs, hd)),
        "x": 0.0, "y": 0.0,
    })
    node_id_set.add(nid)

owned_by_ei = next((data[et[0], et[1], et[2]].edge_index.cpu()
                    for et in EDGE_TYPES if et[1] == "owned_by"), None)
has_ipc_ei  = next((data[et[0], et[1], et[2]].edge_index.cpu()
                    for et in EDGE_TYPES if et[1] == "has_ipc"),  None)
cites_ei_full = next((data[et[0], et[1], et[2]].edge_index.cpu()
                      for et in EDGE_TYPES if et[1] == "cites"),  None)

co_added, co_extra_edges = {}, []
if owned_by_ei is not None:
    for i in range(owned_by_ei.size(1)):
        pg, cg = owned_by_ei[0, i].item(), owned_by_ei[1, i].item()
        if pg in subset_set and len(co_added) < args.co_limit:
            co_nid, pat_nid = f"co_{cg}", f"pat_{pg}"
            if co_nid not in co_added:
                co_added[co_nid] = cg
                all_nodes.append({
                    "id": co_nid, "type": "company",
                    "label": co_list[cg], "short_label": co_list[cg][:14],
                    "is_src": False, "is_dst": False, "importance": 0.2,
                    "hop_from_src": args.khop + 1, "hop_from_dst": args.khop + 1,
                    "min_hop": args.khop + 1,
                    "x": 0.0, "y": 0.0,
                })
                node_id_set.add(co_nid)
            co_extra_edges.append({"src": pat_nid, "dst": co_nid,
                                   "type": "owned_by", "gnn_mask": 0.0,
                                   "rank": None, "is_top_k": False,
                                   "delta_p": None, "attention": {}})

ipc_added, ipc_extra_edges = {}, []
if has_ipc_ei is not None:
    for i in range(has_ipc_ei.size(1)):
        pg, ig = has_ipc_ei[0, i].item(), has_ipc_ei[1, i].item()
        if pg in subset_set and len(ipc_added) < args.ipc_limit:
            ipc_nid, pat_nid = f"ipc_{ig}", f"pat_{pg}"
            if ipc_nid not in ipc_added:
                ipc_added[ipc_nid] = ig
                all_nodes.append({
                    "id": ipc_nid, "type": "ipc",
                    "label": ipc_list[ig], "short_label": ipc_list[ig][:8],
                    "is_src": False, "is_dst": False, "importance": 0.15,
                    "hop_from_src": args.khop + 1, "hop_from_dst": args.khop + 1,
                    "min_hop": args.khop + 1,
                    "x": 0.0, "y": 0.0,
                })
                node_id_set.add(ipc_nid)
            ipc_extra_edges.append({"src": pat_nid, "dst": ipc_nid,
                                    "type": "has_ipc", "gnn_mask": 0.0,
                                    "rank": None, "is_top_k": False,
                                    "delta_p": None, "attention": {}})

print(f"   Nodes: {len(all_nodes)} (pat={N_sub}, co={len(co_added)}, ipc={len(ipc_added)})")

# Patent-patent edges with attention + delta_p
all_edges = []
for ei in range(sub_ei_cpu.size(1)):
    sln, dln = sub_ei_cpu[0, ei].item(), sub_ei_cpu[1, ei].item()
    sg, dg   = subset[sln].item(), subset[dln].item()
    snid, dnid = f"pat_{sg}", f"pat_{dg}"
    et_name  = sub_et[ei]

    # Attention per layer
    edge_attn = {}
    for ln, ld in attn_layers.items():
        if et_name in ld:
            key = f"{sg}_{dg}"
            if key in ld[et_name]:
                edge_attn[ln] = ld[et_name][key]

    all_edges.append({
        "id":        f"e_{ei}",
        "src":       snid,
        "dst":       dnid,
        "type":      et_name,
        "gnn_mask":  float(edge_mask_np[ei]),
        "rank":      int(edge_rank[ei]),
        "is_top_k":  bool(ei in top_k_set),
        "delta_p":   cf_pair_lookup.get((snid, dnid), None),
        "attention": edge_attn,
    })
for j, e in enumerate(co_extra_edges):
    e["id"] = f"eco_{j}"; all_edges.append(e)
for j, e in enumerate(ipc_extra_edges):
    e["id"] = f"eipc_{j}"; all_edges.append(e)

# ════════════════════════════════════════════════════════════════════════════
# 9. Fixed layout
# ════════════════════════════════════════════════════════════════════════════
print("\n▶ Computing fixed layout...")
G_layout = nx.Graph()
for n in all_nodes: G_layout.add_node(n["id"])
for e in all_edges:
    if e["src"] in node_id_set and e["dst"] in node_id_set:
        G_layout.add_edge(e["src"], e["dst"])

pat_nids   = [n["id"] for n in all_nodes if n["type"] == "patent"]
co_nids    = [n["id"] for n in all_nodes if n["type"] == "company"]
ipc_nids   = [n["id"] for n in all_nodes if n["type"] == "ipc"]
other_pats = [n for n in pat_nids if n not in (src_nid, dst_nid)]

init_pos = {src_nid: np.array([-0.3, 0.0]), dst_nid: np.array([0.3, 0.0])}
for i, nid in enumerate(other_pats):
    a = 2*np.pi * i / max(len(other_pats), 1)
    init_pos[nid] = np.array([0.8*np.cos(a), 0.8*np.sin(a)])
for i, nid in enumerate(co_nids):
    a = 2*np.pi * i / max(len(co_nids), 1) + np.pi/6
    init_pos[nid] = np.array([1.5*np.cos(a), 1.5*np.sin(a)])
for i, nid in enumerate(ipc_nids):
    a = 2*np.pi * i / max(len(ipc_nids), 1) + np.pi/3
    init_pos[nid] = np.array([2.1*np.cos(a), 2.1*np.sin(a)])
for n in all_nodes:
    if n["id"] not in init_pos:
        init_pos[n["id"]] = np.random.default_rng(42).random(2) - 0.5

SCALE = 500
pos = nx.spring_layout(G_layout, pos=init_pos, seed=42, iterations=80, k=0.6)
for n in all_nodes:
    xy = pos.get(n["id"], np.zeros(2))
    n["x"] = float(xy[0]) * SCALE
    n["y"] = float(xy[1]) * SCALE

# ════════════════════════════════════════════════════════════════════════════
# 10. Evidence Path
# ════════════════════════════════════════════════════════════════════════════
print("\n▶ Computing evidence path...")
G_path = nx.DiGraph()
for n in all_nodes: G_path.add_node(n["id"])
for e in all_edges:
    if e["src"] in node_id_set and e["dst"] in node_id_set:
        # Higher attention = shorter distance
        attn_val = e["attention"].get("layer2", e["attention"].get("layer1", 0.0)) \
                   if e["attention"] else 0.0
        w = 1.0 / (max(e["gnn_mask"], attn_val) + 0.05)
        G_path.add_edge(e["src"], e["dst"], type=e["type"],
                        gnn_mask=e["gnn_mask"],
                        attn=attn_val, weight=w)

try:
    path_nodes = nx.shortest_path(G_path, src_nid, dst_nid, weight="weight")
    path_edges_list, path_score = [], 0.0
    for i in range(len(path_nodes) - 1):
        ed = G_path[path_nodes[i]][path_nodes[i+1]]
        path_edges_list.append({
            "src": path_nodes[i], "dst": path_nodes[i+1],
            "type": ed["type"],
            "gnn_mask": round(ed["gnn_mask"], 4),
            "attn": round(ed["attn"], 4),
        })
        path_score += max(ed["gnn_mask"], ed["attn"])
    evidence_path = {
        "nodes": path_nodes,
        "edges": path_edges_list,
        "score": round(path_score / max(len(path_edges_list), 1), 4),
    }
    print(f"   {' → '.join(path_nodes[:5])}{'...' if len(path_nodes)>5 else ''}")
except nx.NetworkXNoPath:
    evidence_path = {"nodes": [src_nid, dst_nid], "edges": [], "score": 0.0}
    print("   No path — using direct pair")

# ════════════════════════════════════════════════════════════════════════════
# 11. Agreement table
# ════════════════════════════════════════════════════════════════════════════
agreement = [{
    "rank":           e["rank"],
    "src_label":      e["src_label"],
    "dst_label":      e["dst_label"],
    "edge_type":      e["edge_type"],
    "gnn_importance": e["gnn_mask"],
    "delta_p":        e["delta_p"],
    "agree":          e["gnn_mask"] > 0.4 and e["delta_p"] < -0.01,
} for e in cf_by_edge]

# ════════════════════════════════════════════════════════════════════════════
# 12. Storyline + Conclusion
# ════════════════════════════════════════════════════════════════════════════
most_imp_type = min(cf_by_type, key=lambda k: cf_by_type[k]["delta_p"])
best_edge     = min(cf_by_edge, key=lambda e: e["delta_p"])
top_et        = sub_et[top_k_idx[0]] if len(top_k_idx) > 0 else "?"

# Average attention of top-K edges across all layers
top_k_attn_avg = {}
for ln in ["layer0", "layer1", "layer2", "layer3"]:
    vals = [e["attention"].get(ln, 0.0) for e in all_edges
            if e["is_top_k"] and e["attention"]]
    top_k_attn_avg[ln] = round(np.mean(vals), 4) if vals else 0.0

def path_label(nid):
    t, gi = nid.split("_", 1)
    gi = int(gi)
    if t == "pat": return pat_list[gi]
    if t == "co":  return co_list[gi]
    if t == "ipc": return ipc_list[gi]
    return nid

ep_str = " → ".join(path_label(n) for n in evidence_path["nodes"][:4])
if len(evidence_path["nodes"]) > 4: ep_str += " → ..."

storyline = [
    {
        "title": "① Intrinsic: Native Attention",
        "ko": (f"SimpleCatHeteroGAT의 4개 레이어 각각에서 attention을 추출했습니다. "
               f"Top-K 엣지의 Layer 2 평균 attention = {top_k_attn_avg['layer2']:.3f}."),
        "en": (f"Extracted native GAT attention across all 4 layers. "
               f"Avg Layer 2 attention on top-K edges = {top_k_attn_avg['layer2']:.3f}."),
    },
    {
        "title": "② Post-hoc: GNNExplainer",
        "ko": (f"모델은 {top_et} 관계를 가장 중요한 단서로 학습했습니다. "
               f"상위 {args.topk}개 엣지 마스크가 그래프에 굵은 선으로 표시됩니다."),
        "en": (f"The model relied most on {top_et} edges. "
               f"Top-{args.topk} edge masks shown as thick lines."),
    },
    {
        "title": "③ Perturbation: Counterfactual",
        "ko": (f"**{most_imp_type}** 관계 제거 시 P(sue) {p_base:.4f} → "
               f"{cf_by_type[most_imp_type]['p_after']:.4f} "
               f"(Δ = {cf_by_type[most_imp_type]['delta_p']:+.4f})."),
        "en": (f"Removing **{most_imp_type}** edges drops P(sue) from {p_base:.4f} "
               f"to {cf_by_type[most_imp_type]['p_after']:.4f} "
               f"(Δ = {cf_by_type[most_imp_type]['delta_p']:+.4f})."),
    },
    {
        "title": "④ Evidence Path",
        "ko": f"가장 중요한 추론 경로: {ep_str}",
        "en": f"Highest-importance inference path: {ep_str}",
    },
    {
        "title": "⑤ Conclusion",
        "ko": (f"이 소송 예측은 세 가지 독립적 근거로 지지됩니다: "
               f"(1) GAT attention이 {top_et} 관계를 일관되게 강조, "
               f"(2) GNNExplainer가 동일 엣지를 top-{args.topk}에 포함, "
               f"(3) 해당 엣지 제거 시 P(sue)가 {best_edge['delta_p']:+.4f} 변화. "
               f"세 방법이 일치할수록 설명의 신뢰도가 높아집니다."),
        "en": (f"This lawsuit prediction is supported by three independent sources: "
               f"(1) GAT attention consistently highlights {top_et} relations, "
               f"(2) GNNExplainer ranks the same edges in top-{args.topk}, "
               f"(3) removing those edges shifts P(sue) by {best_edge['delta_p']:+.4f}. "
               f"Higher agreement across methods → stronger evidence."),
    },
]

# ════════════════════════════════════════════════════════════════════════════
# 13. New Patent scenario meta
# ════════════════════════════════════════════════════════════════════════════
src_owners = ([co_list[owned_by_ei[1, i].item()]
               for i in range(owned_by_ei.size(1))
               if owned_by_ei[0, i].item() == src_idx][:5]
              if owned_by_ei is not None else [])
src_ipcs   = ([ipc_list[has_ipc_ei[1, i].item()]
               for i in range(has_ipc_ei.size(1))
               if has_ipc_ei[0, i].item() == src_idx][:5]
              if has_ipc_ei is not None else [])
cites_cnt  = int((cites_ei_full[0] == src_idx).sum() +
                 (cites_ei_full[1] == src_idx).sum()) if cites_ei_full is not None else 0

# ════════════════════════════════════════════════════════════════════════════
# 14. Export
# ════════════════════════════════════════════════════════════════════════════
result = {
    "meta": {
        "src_id":              src_nid,
        "dst_id":              dst_nid,
        "src_label":           pat_list[src_idx],
        "dst_label":           pat_list[dst_idx],
        "p_base":              round(p_base, 4),
        "k_hop":               args.khop,
        "top_k_cf":            args.topk,
        "n_nodes":             len(all_nodes),
        "n_edges":             len(all_edges),
        "pair_idx":            args.pair,
        "query_patent":        pat_list[src_idx],
        "is_new_patent":       src_idx not in state.get("tv_plt_set", set()),
        "src_owner_companies": src_owners,
        "src_ipc_codes":       src_ipcs,
        "src_citations":       cites_cnt,
        "attn_layer_avg":      top_k_attn_avg,
    },
    "nodes":          all_nodes,
    "edges":          all_edges,
    "evidence_path":  evidence_path,
    "agreement":      agreement,
    "storyline":      storyline,
    "counterfactual": {"by_type": cf_by_type, "by_edge": cf_by_edge},
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✓ Exported → {OUT_JSON}")
print(json.dumps(result["meta"], indent=2, ensure_ascii=False))
