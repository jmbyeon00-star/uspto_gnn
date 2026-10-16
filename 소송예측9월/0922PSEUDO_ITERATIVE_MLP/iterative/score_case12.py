from __future__ import annotations
import argparse, gc, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from data import EDGE_TYPES, StaticFeatureStore, load_snapshot, build_features
from model import Model

RUNS_DIR = ROOT / "runs"

DATA_ROOT = Path(
    "/home2/jmbyeon/developments/workspace/apps/uspto_litigation/"
    "소송예측모델 개발/data/with_counter"
)
USABLE_PATH = DATA_ROOT / "gnn_usable_cases_v2.parquet"
CASE_PATENT_PATH = DATA_ROOT / "gnn_case_patents_v2.parquet"
OWNERSHIP_PATH = DATA_ROOT / "ownership_chain_v2.parquet"


def norm(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def make_model(feats):
    dims = {k: int(v.shape[1]) for k, v in feats.items()}
    return Model(dims, config.HIDDEN, EDGE_TYPES, config.DROPOUT)


def pair(r):
    return int(r["src"]), int(r["dst"])


def edge_dict(snapshot):
    d = snapshot["data"]
    return {e: d[e].edge_index.cpu() for e in EDGE_TYPES if e in d.edge_types}


def add_mp(ed, records):
    """train_round.py와 동일하게 historical sue + 누적 pseudo MP를 unique union."""
    et = ("patent", "sue", "patent")
    old = ed.get(et, torch.empty((2, 0), dtype=torch.long)).cpu()
    old_unique = (
        torch.unique(old.t(), dim=0).t().contiguous()
        if old.numel() else old.clone()
    )

    pairs = sorted({pair(r) for r in records})
    add = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs else torch.empty((2, 0), dtype=torch.long)
    )

    merged = torch.cat([old_unique, add], dim=1)
    if merged.numel():
        merged = torch.unique(merged.t(), dim=0).t().contiguous()

    ed[et] = merged
    return {
        "historical_raw": int(old.shape[1]),
        "historical_unique": int(old_unique.shape[1]),
        "pseudo_mp": len(pairs),
        "final_unique": int(merged.shape[1]),
    }


def load_tables():
    for p in [USABLE_PATH, CASE_PATENT_PATH, OWNERSHIP_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)

    usable = pd.read_parquet(USABLE_PATH).copy()
    cp = pd.read_parquet(CASE_PATENT_PATH).copy()
    oc = pd.read_parquet(OWNERSHIP_PATH).copy()

    if "patent_id" in cp.columns and "patent" not in cp.columns:
        cp = cp.rename(columns={"patent_id": "patent"})
    if "patent" in oc.columns and "patent_id" not in oc.columns:
        oc = oc.rename(columns={"patent": "patent_id"})
    if "assignee" in oc.columns and "assignee_id" not in oc.columns:
        oc = oc.rename(columns={"assignee": "assignee_id"})

    usable["date_filed"] = pd.to_datetime(usable["date_filed"], errors="coerce")
    cp["patent"] = cp["patent"].map(norm)
    cp["role"] = cp["role"].astype(str).str.lower()
    oc["patent_id"] = oc["patent_id"].map(norm)
    oc["assignee_id"] = oc["assignee_id"].map(norm)
    oc["start_dt"] = pd.to_datetime(oc["start_dt"], errors="coerce")
    oc["end_dt"] = pd.to_datetime(oc["end_dt"], errors="coerce")

    case_patents = {}
    for case_id, g in cp.groupby("case_row_id"):
        case_patents[int(case_id)] = {
            "plaintiff": set(g.loc[g["role"].eq("plaintiff"), "patent"].dropna()),
            "defendant": set(g.loc[g["role"].eq("defendant"), "patent"].dropna()),
        }

    ownership = {
        str(a): g for a, g in oc.groupby("assignee_id", sort=False)
    }
    return usable, case_patents, ownership


def get_portfolio(groups, assignee_id, event_dt):
    """반드시 사건 발생일 당시 active portfolio."""
    aid = norm(assignee_id)
    if aid is None or aid not in groups or pd.isna(event_dt):
        return []

    g = groups[aid]
    q = g[
        (g["start_dt"].isna() | g["start_dt"].le(event_dt))
        & (g["end_dt"].isna() | g["end_dt"].gt(event_dt))
    ]
    return sorted(set(q["patent_id"].dropna().astype(str)))


@torch.no_grad()
def score_batch(model, z, src, dst, device, batch_size):
    logits = []
    for i in range(0, len(src), batch_size):
        s = torch.tensor(src[i:i + batch_size], dtype=torch.long, device=device)
        d = torch.tensor(dst[i:i + batch_size], dtype=torch.long, device=device)
        logits.append(model.score(z, s, d).detach().cpu())
    logit = torch.cat(logits).numpy()
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
    return logit, prob


def run(cutoff, target_start, target_end, round_idx, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rdir = RUNS_DIR / f"cutoff_{cutoff}" / f"round_{round_idx:02d}"
    rpath = rdir / "round_data.pkl"
    ck = rdir / "checkpoint.pt"

    if not rpath.exists():
        raise FileNotFoundError(f"round_data 없음: {rpath}")
    if not ck.exists():
        raise FileNotFoundError(f"checkpoint 없음: {ck}")

    import pickle
    with open(rpath, "rb") as f:
        rd = pickle.load(f)

    snap = load_snapshot(config.SNAP_DIR, cutoff)
    p2i = {str(k): int(v) for k, v in snap["pat2idx"].items()}
    i2p = {int(v): str(k) for k, v in snap["pat2idx"].items()}

    static = StaticFeatureStore.load(config.DATA_GNN)
    feats, _ = build_features(snap, static)
    ed = edge_dict(snap)
    mp = add_mp(ed, rd.get("mp_pseudo", []))

    counts = {k: int(v.shape[0]) for k, v in feats.items()}
    fd = {k: v.to(device) for k, v in feats.items()}
    gd = {
        "edge_dict": {k: v.to(device) for k, v in ed.items()},
        "counts": counts,
    }

    model = make_model(feats).to(device)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        z = model.encode(fd, gd)

    usable, case_patents, ownership = load_tables()

    # 이 fold에서 pseudo로 사용할 수 있는 사건:
    # historical(<=cutoff) -> 이후 MP 후보
    # target_start~target_end -> 이후 SUP 후보
    end_dt = pd.Timestamp(f"{target_end}-12-31")
    cases = usable[
        usable["case_type"].astype(str).isin(["case1", "case2"])
        & usable["date_filed"].notna()
        & usable["date_filed"].le(end_dt)
    ].copy()

    rows = []
    fail = {
        "no_known": 0,
        "known_not_in_snapshot": 0,
        "no_portfolio": 0,
        "candidate_not_in_snapshot": 0,
    }

    print("=" * 100)
    print(f"SCORE CASE1/2 | cutoff={cutoff} target={target_start}-{target_end} round={round_idx}")
    print(f"checkpoint : {ck}")
    print(f"cases      : {len(cases):,}")
    print(f"embedding  : {tuple(z.shape)}")
    print(
        f"sue graph  : {mp['historical_raw']:,} raw / "
        f"{mp['historical_unique']:,} historical unique + "
        f"{mp['pseudo_mp']:,} pseudo MP -> {mp['final_unique']:,} final unique"
    )
    print("=" * 100)

    for pos, (_, c) in enumerate(cases.iterrows(), 1):
        case_id = int(c["case_row_id"])
        ctype = str(c["case_type"])
        event_dt = c["date_filed"]
        pats = case_patents.get(case_id, {})

        if ctype == "case1":
            known_patents = sorted(pats.get("plaintiff", set()))
            company_id = c.get("def_assignee_id")
            missing_side = "defendant"
        else:
            known_patents = sorted(pats.get("defendant", set()))
            company_id = c.get("plt_assignee_id")
            missing_side = "plaintiff"

        if not known_patents:
            fail["no_known"] += 1
            continue

        known_in = [(p, p2i[p]) for p in known_patents if p in p2i]
        if not known_in:
            fail["known_not_in_snapshot"] += 1
            continue

        portfolio_raw = get_portfolio(ownership, company_id, event_dt)
        if not portfolio_raw:
            fail["no_portfolio"] += 1
            continue

        known_set = {p for p, _ in known_in}
        candidate_patents = [
            p for p in portfolio_raw if p in p2i and p not in known_set
        ]
        candidate_patents = sorted(set(candidate_patents))
        if not candidate_patents:
            fail["candidate_not_in_snapshot"] += 1
            continue

        candidate_idx = [p2i[p] for p in candidate_patents]

        # known patent가 여러 개면 각각의 query를 모두 보존.
        # 따라서 전체 candidate score 분포를 잃지 않는다.
        for known_patent, known_idx in known_in:
            if ctype == "case1":
                src = [known_idx] * len(candidate_idx)
                dst = candidate_idx
            else:
                src = candidate_idx
                dst = [known_idx] * len(candidate_idx)

            logits, probs = score_batch(
                model, z, src, dst, device, batch_size
            )
            order = np.argsort(-probs)
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1)

            for j, cand_patent in enumerate(candidate_patents):
                if ctype == "case1":
                    plt_patent, def_patent = known_patent, cand_patent
                    plt_idx, def_idx = known_idx, candidate_idx[j]
                else:
                    plt_patent, def_patent = cand_patent, known_patent
                    plt_idx, def_idx = candidate_idx[j], known_idx

                rows.append({
                    "cutoff": cutoff,
                    "target_start": target_start,
                    "target_end": target_end,
                    "round": round_idx,
                    "case_row_id": case_id,
                    "case_type": ctype,
                    "date_filed": event_dt,
                    "event_year": int(event_dt.year),
                    "pseudo_bucket": (
                        "mp" if event_dt.year <= cutoff
                        else "sup" if target_start <= event_dt.year <= target_end
                        else "gap"
                    ),
                    "missing_side": missing_side,
                    "known_patent": known_patent,
                    "known_idx": known_idx,
                    "candidate_patent": cand_patent,
                    "candidate_idx": candidate_idx[j],
                    "candidate_count": len(candidate_idx),
                    "plt_patent": plt_patent,
                    "def_patent": def_patent,
                    "src": int(plt_idx),
                    "dst": int(def_idx),
                    "logit": float(logits[j]),
                    "score": float(probs[j]),
                    "rank": int(ranks[j]),
                })

        if pos % 500 == 0:
            print(f"{pos:,}/{len(cases):,} cases | candidate rows={len(rows):,}")

    out = pd.DataFrame(rows)
    out_path = rdir / "scored_candidates.parquet"
    out.to_parquet(out_path, index=False)

    print("\n" + "=" * 100)
    print("SCORING COMPLETE")
    print("=" * 100)
    print("candidate rows :", f"{len(out):,}")
    print("failures       :", fail)
    if len(out):
        print("\nSCORE DISTRIBUTION (ALL CANDIDATE PAIRS)")
        print(out["score"].describe(
            percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .975, .99, .995]
        ).to_string())
        print("\nBUCKET")
        print(out["pseudo_bucket"].value_counts().to_string())
        print("\nCASE TYPE")
        print(out["case_type"].value_counts().to_string())
    print("\nsaved:", out_path)

    del model, fd, gd, z
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--target-start", type=int, required=True)
    ap.add_argument("--target-end", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=65536)
    a = ap.parse_args()

    run(
        a.cutoff,
        a.target_start,
        a.target_end,
        a.round,
        a.batch_size,
    )
