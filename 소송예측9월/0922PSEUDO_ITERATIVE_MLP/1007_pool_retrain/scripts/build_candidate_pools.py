from __future__ import annotations
import pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR.parent
ROOT = EXP_DIR.parent
sys.path.insert(0, str(ROOT))
import config
from data import load_snapshot

CUTOFFS = [2004, 2006, 2008, 2010]
TOPK = 2000
EMB_DIR = EXP_DIR / "embeddings(snapshot등장모든특허)"
EMB_PATH = EMB_DIR / "embeddings.npy"
META_PATH = EMB_DIR / "embedding_meta.parquet"
SOURCE_DIR = EXP_DIR / "source_state"
OUT_DIR = EXP_DIR / "candidate_pools"

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def save_pkl(x, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

def pair(r):
    return int(r["src"]), int(r["dst"])

def ipc_prefix(code, n):
    if code is None:
        return None
    s = str(code).strip().upper().replace(" ", "")
    return s[:n] if len(s) >= n else s

def build_maps(snap):
    data = snap["data"]
    p3, i3p = defaultdict(set), defaultdict(set)
    p4, i4p = defaultdict(set), defaultdict(set)
    adj = defaultdict(set)
    ipc_list = [str(x) for x in snap.get("ipc_list", [])]

    et = ("patent", "has_ipc", "ipc")
    if et in data.edge_types:
        for p, i in data[et].edge_index.cpu().t().tolist():
            p, i = int(p), int(i)
            if i >= len(ipc_list):
                continue
            c3 = ipc_prefix(ipc_list[i], 3)
            c4 = ipc_prefix(ipc_list[i], 4)
            if c3:
                p3[p].add(c3); i3p[c3].add(p)
            if c4:
                p4[p].add(c4); i4p[c4].add(p)

    for et in [
        ("patent", "cites", "patent"),
        ("patent", "cited_by", "patent"),
    ]:
        if et not in data.edge_types:
            continue
        for s, t in data[et].edge_index.cpu().t().tolist():
            s, t = int(s), int(t)
            adj[s].add(t); adj[t].add(s)

    return p3, i3p, p4, i4p, adj

def same_ipc(src, p2ipc, ipc2pat):
    out = set()
    for c in p2ipc.get(src, set()):
        out |= ipc2pat.get(c, set())
    out.discard(src)
    return out

def citation_layers(adj, src):
    seen, frontier, exact = {src}, {src}, []
    for _ in range(3):
        nxt = set()
        for x in frontier:
            nxt |= adj.get(x, set())
        nxt -= seen
        exact.append(nxt)
        seen |= nxt
        frontier = nxt
    c1 = exact[0]
    c2 = c1 | exact[1]
    c3 = c2 | exact[2]
    return c1, c2, c3

def required_srcs(rd):
    srcs = set()
    for key in ["sup_all", "val_fixed"]:
        for r in rd.get(key, []):
            srcs.add(pair(r)[0])
    return sorted(srcs)

def embedding_map(meta):
    if "patent_id" not in meta.columns:
        raise RuntimeError("embedding_meta.parquet에 patent_id 컬럼이 없습니다.")
    return {
        str(pid): int(row)
        for row, pid in enumerate(meta["patent_id"])
    }

def build_one(cutoff, emb, patent_to_row):
    print("\n" + "=" * 70)
    print("CUTOFF:", cutoff)
    print("=" * 70)

    snap = load_snapshot(config.SNAP_DIR, cutoff)
    pat2idx = {str(k): int(v) for k, v in snap["pat2idx"].items()}
    idx2pat = {v: k for k, v in pat2idx.items()}

    state = SOURCE_DIR / f"cutoff_{cutoff}" / "round_data.pkl"
    rd = load_pkl(state)
    srcs = required_srcs(rd)
    print("required src:", len(srcs))

    p3, i3p, p4, i4p, adj = build_maps(snap)
    pools, sizes = {}, []
    missing_src = []
    missing_cand = 0

    for i, src in enumerate(srcs, 1):
        patent_id = idx2pat.get(src)
        if patent_id is None:
            raise RuntimeError(f"snapshot에 src 없음: {src}")

        src_row = patent_to_row.get(patent_id)
        if src_row is None:
            missing_src.append(src)
            continue

        ipc3 = same_ipc(src, p3, i3p)
        ipc4 = same_ipc(src, p4, i4p)
        c1, c2, c3 = citation_layers(adj, src)

        base = (
            (ipc3 & c1)
            | (ipc3 & c2)
            | (ipc3 & c3)
            | c1
            | c2
        )
        base.discard(src)

        nodes, rows = [], []
        for node in ipc4 - base:
            pid = idx2pat.get(int(node))
            row = patent_to_row.get(pid) if pid is not None else None
            if row is None:
                missing_cand += 1
                continue
            nodes.append(int(node))
            rows.append(int(row))

        if nodes:
            rows = np.asarray(rows, dtype=np.int64)
            src_vec = np.asarray(emb[src_row], dtype=np.float32)
            scores = np.asarray(emb[rows] @ src_vec)
            order = np.argsort(scores)[::-1][:TOPK]
            ipc4_top = {nodes[j] for j in order}
        else:
            ipc4_top = set()

        raw = set(base) | ipc4_top
        raw.discard(src)
        pools[src] = sorted(raw)
        sizes.append(len(raw))

        if i % 50 == 0 or i == len(srcs):
            print(f"[{i}/{len(srcs)}] src={src} | pool={len(raw):,}")

    if missing_src:
        raise RuntimeError(f"embedding 없는 SRC: {missing_src[:20]}")

    payload = {
        "cutoff": cutoff,
        "topk": TOPK,
        "policy": "Base(A1+A2+A3+citation_1+citation_2) + full IPC4 semantic Top2000",
        "source_state": str(state),
        "src_policy": "sup_all + val_fixed unique SRC",
        "raw_pools_by_src": pools,
        "raw_pool_stats": {
            "src": len(pools),
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
        },
        "missing_candidate_embedding": int(missing_cand),
    }

    out = OUT_DIR / f"cutoff_{cutoff}.pkl"
    save_pkl(payload, out)

    s = payload["raw_pool_stats"]
    print(
        "mean/median/max:",
        f"{s['mean']:,.1f} / {s['median']:,.0f} / {s['max']:,}",
    )
    print("saved:", out)

def main():
    print("[1] Load embeddings")
    emb = np.load(EMB_PATH, mmap_mode="r")
    meta = pd.read_parquet(META_PATH)

    if len(meta) != len(emb):
        raise RuntimeError("embedding/meta 길이가 다릅니다.")

    patent_to_row = embedding_map(meta)
    print("embedding:", emb.shape)
    print("meta:", meta.shape)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cutoff in CUTOFFS:
        build_one(cutoff, emb, patent_to_row)

    print("\nDONE")

if __name__ == "__main__":
    main()
