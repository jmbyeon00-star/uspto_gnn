"""components/graph.py — PyVis graph builder (reusable across all pages)."""
from __future__ import annotations
import os, tempfile
from pyvis.network import Network
from config import NODE_COLORS, EDGE_COLORS, PYVIS_OPTIONS


def _node_style(n, prediction_mode=False, highlight_ids=None, path_ids=None):
    nid   = n["id"]
    ntype = n["type"]

    if n.get("is_src"):
        color, shape, size_mult, border = NODE_COLORS["patent_src"], "star",    2.0, "#ff4b4b"
    elif prediction_mode and n.get("is_predicted_gt"):
        color, shape, size_mult, border = NODE_COLORS["patent_predicted_gt"], "diamond", 2.0, "#f1c40f"
    elif prediction_mode and n.get("is_predicted"):
        color, shape, size_mult, border = NODE_COLORS["patent_predicted"], "dot", 0.9, NODE_COLORS["patent_predicted"]
    elif n.get("is_dst"):
        color, shape, size_mult, border = NODE_COLORS["patent_dst"], "diamond", 1.7, "#1e90ff"
    elif path_ids and nid in path_ids:
        color, shape, size_mult, border = NODE_COLORS["path"],       "diamond", 1.3, "#f1c40f"
    elif ntype == "company":
        color, shape, size_mult, border = NODE_COLORS["company"],    "triangle",1.0, NODE_COLORS["company"]
    elif ntype == "ipc":
        color, shape, size_mult, border = NODE_COLORS["ipc"],        "hexagon", 0.8, NODE_COLORS["ipc"]
    else:
        color, shape, size_mult, border = NODE_COLORS["patent"],     "dot",     1.0, NODE_COLORS["patent"]

    if highlight_ids and nid in highlight_ids:
        border = "#ffffff"
        size_mult *= 1.3

    return color, shape, size_mult, border


def _edge_style(e, top_k_ids=None, path_pairs=None, mode="importance",
                attn_layer=None, prediction_mode=False):
    eid  = e["id"]
    pair = (e["source"], e["target"])

    on_path = path_pairs and pair in path_pairs
    is_top  = top_k_ids and eid in top_k_ids

    # Predicted edges override all other styling
    if e.get("is_predicted"):
        prob = e.get("probability", e.get("importance", 0.3))
        if e.get("is_ground_truth"):
            return "#f1c40f", max(3.0, prob * 12), 0.95, False   # Gold, solid, thick
        else:
            return "#c678dd", max(1.0, prob * 8),  0.60, True    # Purple, dashed

    # In prediction mode, context edges are faded to let predictions stand out
    if prediction_mode:
        color   = "#444444"
        width   = 1.0
        opacity = 0.18
        dashes  = False
        return color, width, opacity, dashes

    if mode == "importance":
        score = e.get("importance", 0.0)
        color = EDGE_COLORS.get(e["relation"], "#888888")
    elif mode.startswith("attn"):
        lk    = attn_layer or "layer2"
        score = (e.get("attention_layers") or {}).get(lk, e.get("attention", 0.0))
        v     = max(0.0, min(1.0, score))
        b     = int(60 + 195 * v)
        color = f"#{30:02x}{80:02x}{b:02x}"
    elif mode == "overlay":
        attn_v = (e.get("attention_layers") or {}).get("layer2", e.get("attention", 0.0))
        gnn_v  = e.get("importance", 0.0)
        cf_v   = 1.0 if e.get("is_counterfactual") else 0.0
        r = int(40 + 200 * gnn_v)
        g = int(40 + 200 * cf_v)
        b = int(40 + 200 * attn_v)
        color = f"#{r:02x}{g:02x}{b:02x}"
        score = max(attn_v, gnn_v, cf_v)
    else:
        score = e.get("importance", 0.0)
        color = EDGE_COLORS.get(e["relation"], "#888888")

    if on_path:
        color = "#f1c40f"

    width   = max(1.0, score * 14)
    opacity = 0.9 if (is_top or on_path) else 0.45
    dashes  = not (is_top or on_path)
    return color, width, opacity, dashes


def _edge_tooltip(e):
    if e.get("is_predicted"):
        gt_marker = " — GROUND TRUTH ✓" if e.get("is_ground_truth") else ""
        return (
            f"<b>Predicted Lawsuit{gt_marker}</b><br>"
            f"Rank: #{e.get('predicted_rank','?')}<br>"
            f"P(sue): {e.get('probability', 0):.4f}"
        )
    attn_layers = e.get("attention_layers") or {}
    attn_str = " | ".join(f"L{i+1}={v:.3f}" for i, (_, v) in enumerate(attn_layers.items())) or "—"
    cf = "✅ Counterfactual" if e.get("is_counterfactual") else ""
    path = "🔗 Evidence Path" if e.get("is_path") else ""
    return (
        f"<b>Relation: {e['relation']}</b><br>"
        f"Raw type: {e['raw_type']}<br>"
        f"Importance: {e.get('importance',0):.4f}<br>"
        f"Attention: {attn_str}<br>"
        + (f"<br>{cf}" if cf else "")
        + (f"<br>{path}" if path else "")
    )


def build_pyvis(
    nodes: list,
    edges: list,
    height: str = "580px",
    mode: str = "importance",      # importance | attn_layer1..4 | overlay | hop
    attn_layer: str = "layer2",
    node_scale: int = 28,
    hop_limit: int = 99,
    show_company: bool = True,
    show_ipc: bool = True,
    threshold: float = 0.0,
    top_k_ids: set | None = None,
    path_node_ids: set | None = None,
    path_edge_pairs: set | None = None,
    highlight_nodes: set | None = None,
    show_predicted: bool = False,   # Show predicted lawsuit edges
    prediction_mode: bool = False,  # Prediction overlay mode (fades context edges)
) -> str:
    """Build PyVis graph and return HTML string."""
    net = Network(height=height, width="100%",
                  bgcolor="#0d1117", font_color="#e6edf3",
                  directed=True, notebook=False)
    net.set_options(PYVIS_OPTIONS)

    # Filter nodes
    ntype_allowed = {"patent"}
    if show_company: ntype_allowed.add("company")
    if show_ipc:     ntype_allowed.add("ipc")

    if mode == "hop":
        visible_nodes = {n["id"] for n in nodes
                         if n["type"] in ntype_allowed
                         and (n.get("min_hop") is None or n["min_hop"] <= hop_limit)}
    else:
        used = set()
        for e in edges:
            if e.get("is_predicted"):
                if show_predicted:
                    used.add(e["source"])
                    used.add(e["target"])
            elif e.get("importance", 1.0) >= threshold or e["relation"] in ("owner", "ipc"):
                used.add(e["source"])
                used.add(e["target"])

        # Always include src/dst
        for n in nodes:
            if n.get("is_src") or n.get("is_dst"):
                used.add(n["id"])

        visible_nodes = {n["id"] for n in nodes
                         if n["id"] in used and n["type"] in ntype_allowed}

    node_map = {n["id"]: n for n in nodes}

    for n in nodes:
        if n["id"] not in visible_nodes:
            continue
        color, shape, size_mult, border = _node_style(
            n, prediction_mode, highlight_nodes, path_node_ids)
        sz = max(6, n.get("importance", 0.3) * node_scale * size_mult)

        hop_s = n.get("hop_from_src", "?")
        hop_d = n.get("hop_from_dst", "?")

        if n.get("is_predicted"):
            is_gt = n.get("is_predicted_gt", False)
            gt_line = "<br>🟡 <b>GROUND TRUTH</b> ✓" if is_gt else ""
            tooltip = (
                f"<b>{n['label']}</b><br>"
                f"Owner: {n.get('owner','—')}<br>"
                f"P(sue): {n.get('embedding_similarity',0):.4f}"
                + gt_line
                + "<br>📌 Predicted defendant"
            )
        else:
            tooltip = (
                f"<b>{n['label']}</b><br>"
                f"Type: {n['type']}<br>"
                f"Owner: {n.get('owner','—')}<br>"
                f"IPC: {', '.join(n.get('ipc',[])[:2]) or '—'}<br>"
                f"Importance: {n.get('importance',0):.3f}<br>"
                f"Emb sim: {n.get('embedding_similarity',0):.3f}<br>"
                f"Hop (src/dst): {hop_s}/{hop_d}"
                + ("<br>⭐ Plaintiff"     if n.get("is_src") else "")
                + ("<br>🛡️ Defendant"    if n.get("is_dst") else "")
                + ("<br>🔗 Evidence Path" if path_node_ids and n["id"] in path_node_ids else "")
            )
        net.add_node(
            n["id"], x=n["x"], y=n["y"], physics=False,
            label=n["short_label"],
            color={"background": color, "border": border},
            shape=shape, size=sz, title=tooltip,
            font={"size": 9, "color": "#e6edf3"},
        )

    for e in edges:
        if e["source"] not in visible_nodes or e["target"] not in visible_nodes:
            continue
        if e.get("is_predicted"):
            if not show_predicted:
                continue
        elif e.get("importance", 1.0) < threshold and e["relation"] not in ("owner", "ipc"):
            continue

        color, width, opacity, dashes = _edge_style(
            e, top_k_ids, path_edge_pairs, mode, attn_layer, prediction_mode)
        net.add_edge(
            e["source"], e["target"],
            width=width, dashes=dashes,
            color={"color": color, "opacity": opacity},
            title=_edge_tooltip(e),
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                     mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        html = open(tmp.name, encoding="utf-8").read()
    os.unlink(tmp.name)
    return html
