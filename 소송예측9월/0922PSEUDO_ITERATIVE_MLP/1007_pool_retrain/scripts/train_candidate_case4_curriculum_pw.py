from __future__ import annotations
import argparse, gc, json, pickle, random, sys
from collections import deque
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

EXP  = Path(__file__).resolve().parent.parent
BASE = EXP.parent
sys.path.insert(0, str(BASE))

CURNM = BASE / "CurNM" / "Ring_dens"
sys.path.insert(0, str(CURNM))
from policy import AdaptiveThresholdPolicy


def ring_boundary(pool_size: int, thres_outer: float, thres_inner: float = 0.0):
    outer_idx = max(round(thres_outer * pool_size), 1)
    inner_idx = round(thres_inner * outer_idx)
    return outer_idx, inner_idx

import config
from data import EDGE_TYPES, StaticFeatureStore, load_snapshot, build_features
from model import Model

FOLDS = [2004, 2006, 2008, 2010]

M             = 8
MIN_RAND_PROP = 0.3
SCORE_WINDOW  = 5
DELTA         = 0.05
THRES_INNER   = 0.0
LOG_SRCS      = 3


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_pkl(p):
    with open(p, "rb") as f: return pickle.load(f)

def pair(r):
    if isinstance(r, dict): return int(r["src"]), int(r["dst"])
    return int(r[0]), int(r[1])

def edge_dict(snapshot):
    d = snapshot["data"]
    return {e: d[e].edge_index.cpu() for e in EDGE_TYPES if e in d.edge_types}

def make_model(feats):
    dims = {k: int(v.shape[1]) for k, v in feats.items()}
    return Model(dims, config.HIDDEN, EDGE_TYPES, config.DROPOUT)

def mask_fixed_val_from_historical_sue(ed, val_records):
    et = ("patent", "sue", "patent")
    val_pairs = {pair(r) for r in val_records}
    old = ed.get(et, torch.empty((2, 0), dtype=torch.long)).cpu()
    if old.numel() == 0:
        print("[LEAKAGE CHECK] historical sue MP ∩ fixed VAL = 0")
        return {"historical_val_mask_raw_removed": 0, "historical_val_mask_unique_pairs": 0}
    keep = []; removed = set()
    for s, d in old.t().tolist():
        p = (int(s), int(d)); keep.append(p not in val_pairs)
        if p in val_pairs: removed.add(p)
    keep = torch.tensor(keep, dtype=torch.bool)
    filtered = old[:, keep].contiguous(); ed[et] = filtered
    remaining = {(int(s), int(d)) for s, d in filtered.t().tolist()}
    if remaining & val_pairs: raise RuntimeError("historical sue MP에 VAL leakage가 남아 있습니다.")
    print(f"[VAL MP MASK] raw removed={int((~keep).sum()):,} | unique={len(removed):,}")
    return {"historical_val_mask_raw_removed": int((~keep).sum()), "historical_val_mask_unique_pairs": len(removed)}

def add_mp(ed, records):
    et = ("patent", "sue", "patent")
    old = ed.get(et, torch.empty((2, 0), dtype=torch.long)).cpu()
    old_u = torch.unique(old.t(), dim=0).t().contiguous() if old.numel() else old.clone()
    pairs = sorted({pair(r) for r in records})
    add = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
    merged = torch.cat([old_u, add], dim=1)
    if merged.numel(): merged = torch.unique(merged.t(), dim=0).t().contiguous()
    ed[et] = merged
    return {"historical_sue_raw": int(old.shape[1]), "historical_sue_unique": int(old_u.shape[1]),
            "pseudo_mp_requested": len(pairs), "actual_new_pseudo_mp": int(merged.shape[1] - old_u.shape[1]),
            "final_sue_unique": int(merged.shape[1])}

@torch.no_grad()
def full_val_metrics(model, z, records, pools, retrieved, device):
    ranks = []; rr_all = []; retrieved_ranks = []
    for r in records:
        s, d = pair(r)
        neg = pools.get(s, pools.get(str(s), []))
        if not neg: raise RuntimeError(f"VAL candidate pool empty: src={s}")
        src = torch.full((1 + len(neg),), s, dtype=torch.long, device=device)
        dst = torch.tensor([d] + [int(x) for x in neg], dtype=torch.long, device=device)
        sc  = model.score(z, src, dst)
        rank = 1 + int((sc[1:] > sc[0]).sum().item())
        ranks.append(rank)
        hit = bool(retrieved.get((s, d), retrieved.get(str((s, d)), False)))
        if hit:
            retrieved_ranks.append(rank); rr_all.append(1.0 / rank)
        else:
            rr_all.append(0.0)
    a  = np.asarray(ranks, dtype=np.int64)
    ar = np.asarray(retrieved_ranks, dtype=np.int64)
    out = {"MRR_end2end": float(np.mean(rr_all)),
           "retrieved": int(len(ar)), "total": int(len(records)),
           "retrieval_recall": float(len(ar) / len(records)),
           "MeanRank_all_scored": float(np.mean(a)),
           "MedianRank_all_scored": float(np.median(a))}
    out["MRR_retrieved"] = float(np.mean(1 / ar)) if len(ar) else 0.0
    for k in [1, 10, 50, 100, 500, 1000]:
        out[f"Hits@{k}_end2end"]   = float(np.mean([(1 if h and rank <= k else 0)
            for rank, h in zip(ranks, [bool(retrieved.get(pair(r), False)) for r in records])]))
        out[f"Hits@{k}_retrieved"] = float(np.mean(ar <= k)) if len(ar) else 0.0
    return out

@torch.no_grad()
def score_pools(model, z, train_pools, device):
    result = {}
    for s_raw, pool in train_pools.items():
        s = int(s_raw)
        if len(pool) == 0:
            result[s] = np.array([])
            continue
        s_t = torch.full((len(pool),), s, dtype=torch.long, device=device)
        d_t = torch.tensor([int(x) for x in pool], dtype=torch.long, device=device)
        result[s] = model.score(z, s_t, d_t).cpu().numpy()
    return result

def ring_sample_neg(pool, score_hist, thres_outer, n_ring, n_rand, rng):
    pool_size = len(pool)
    if score_hist:
        stable = np.nanmean(np.stack(list(score_hist), axis=0), axis=0)
    else:
        stable = np.zeros(pool_size)
    outer_idx, inner_idx = ring_boundary(pool_size, thres_outer, THRES_INNER)
    outer_idx = max(outer_idx, n_ring)
    sorted_idx = np.argsort(stable)[::-1]
    ring_zone  = sorted_idx[inner_idx:outer_idx]
    if len(ring_zone) >= n_ring:
        chosen_ring = rng.choice(ring_zone, size=n_ring, replace=False)
    else:
        extra = rng.choice(
            np.setdiff1d(np.arange(pool_size), ring_zone),
            size=n_ring - len(ring_zone), replace=False,
        )
        chosen_ring = np.concatenate([ring_zone, extra])
    chosen_rand = rng.choice(pool_size, size=n_rand, replace=False)
    ring_neg = [int(pool[i]) for i in chosen_ring]
    rand_neg = [int(pool[i]) for i in chosen_rand]
    return ring_neg, rand_neg, int(outer_idx), int(inner_idx)


def run_fold(cutoff, pseudo_w, out_dir, warm=None):
    seed_all(config.SEED + cutoff)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_path   = EXP / "source_state" / f"cutoff_{cutoff}" / "round_data.pkl"
    fixed_path = EXP / "fixed_data"   / f"cutoff_{cutoff}" / "fixed_data.pkl"
    if not src_path.exists():   raise FileNotFoundError(src_path)
    if not fixed_path.exists(): raise FileNotFoundError(fixed_path)

    fx = load_pkl(fixed_path)
    tr = fx["sup_all"]; va = fx["val_fixed"]
    train_pools = fx["train_negative_pools"]; val_pools = fx["val_candidate_pools"]
    retrieved   = fx["val_retrieved"]

    snap   = load_snapshot(config.SNAP_DIR, cutoff)
    static = StaticFeatureStore.load(config.DATA_GNN)
    feats, info = build_features(snap, static); ed = edge_dict(snap)
    mask_stats = mask_fixed_val_from_historical_sue(ed, va)
    mp_stats   = add_mp(ed, fx.get("mp_pseudo", [])); mp_stats.update(mask_stats)

    counts = {k: int(v.shape[0]) for k, v in feats.items()}
    fd = {k: v.to(device) for k, v in feats.items()}
    gd = {"edge_dict": {k: v.to(device) for k, v in ed.items()}, "counts": counts}

    # ── GT / Pseudo 구분 weight 벡터 ─────────────────────────────────────────
    pseudo_set = {pair(r) for r in fx.get("sup_pseudo", [])}
    train_pairs = [pair(r) for r in tr]
    pos_w = torch.tensor(
        [pseudo_w if p in pseudo_set else 1.0 for p in train_pairs],
        dtype=torch.float, device=device,
    )
    gt_count     = int((pos_w == 1.0).sum().item())
    pseudo_count = len(train_pairs) - gt_count

    val_pairs = {pair(r) for r in va}
    if len(train_pairs) != len(set(train_pairs)): raise RuntimeError("SUP ALL 중복 pair")
    if set(train_pairs) & val_pairs:              raise RuntimeError("SUP/VAL overlap")

    model = make_model(feats).to(device)
    if warm:
        model.load_state_dict(torch.load(warm, map_location="cpu", weights_only=True))
        print("warm start <-", warm)
    else:
        print("fresh init (2004)")

    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    rng = np.random.default_rng(config.SEED + 100000 + cutoff)
    ps  = torch.tensor([s for s, _ in train_pairs], dtype=torch.long, device=device)
    pd  = torch.tensor([d for _, d in train_pairs], dtype=torch.long, device=device)

    thres_policy = AdaptiveThresholdPolicy(
        total_steps=config.EPOCHS,
        lower_is_better=False,
        init_thres=1.0, min_thres=0.1, max_thres=1.0,
        delta=DELTA, window=3,
    )
    score_history: dict[int, deque] = {}
    log_srcs = [int(s) for s, _ in train_pairs[:LOG_SRCS]]
    prev_ring_sets: dict[int, set] = {}

    out = out_dir / f"cutoff_{cutoff}"; out.mkdir(parents=True, exist_ok=True)
    ring_log_f = open(out / "ring_log.jsonl", "w")

    print("=" * 90)
    print(f"CASE4 CurNM+PW | cutoff={cutoff} | SUP={len(tr):,} (GT={gt_count:,} w=1.0 | Pseudo={pseudo_count:,} w={pseudo_w})")
    print(f"TRAIN NEG = Ring {M} + Random {M} per positive | total={2*M} | BCE Loss")
    print(f"CurNM: WINDOW={SCORE_WINDOW} | DELTA={DELTA} | INNER={THRES_INNER} | MIN_RAND={MIN_RAND_PROP}")
    print(f"VAL = Full Pool Ranking | retrieval={sum(retrieved.values()):,}/{len(retrieved):,}")
    print(f"max epochs={config.EPOCHS} | val every={config.VAL_EVERY} | patience={config.PATIENCE}")
    print("=" * 90)

    best = -1.; best_epoch = 0; best_state = None; best_vm = None; patience = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        z = model.encode(fd, gd)

        with torch.no_grad():
            cur_scores = score_pools(model, z, train_pools, device)
        for s_raw in train_pools:
            s = int(s_raw)
            if s not in score_history:
                score_history[s] = deque(maxlen=SCORE_WINDOW)
            score_history[s].append(cur_scores[s])

        thres_outer = thres_policy.get_threshold(epoch - 1)
        rand_prop   = max(thres_outer, MIN_RAND_PROP)

        ring_ns, ring_nd, rand_ns, rand_nd = [], [], [], []
        epoch_ring_log = {}
        for s, _ in train_pairs:
            pool = train_pools.get(s, train_pools.get(str(s), []))
            hist = score_history.get(s, deque())
            ring_neg, rand_neg, outer_idx, inner_idx = ring_sample_neg(
                pool, hist, thres_outer, M, M, rng
            )
            ring_ns.extend([s] * M); ring_nd.extend(ring_neg)
            rand_ns.extend([s] * M); rand_nd.extend(rand_neg)

            if s in log_srcs:
                cur_set  = set(ring_neg)
                prev_set = prev_ring_sets.get(s, set())
                overlap  = len(cur_set & prev_set) if prev_set else None
                epoch_ring_log[str(s)] = {
                    "ring_sample": sorted(ring_neg),
                    "overlap_with_prev": overlap,
                    "overlap_ratio": round(overlap / M, 3) if overlap is not None else None,
                    "outer_idx": outer_idx, "inner_idx": inner_idx, "pool_size": len(pool),
                }
                prev_ring_sets[s] = cur_set

        ring_ns_t = torch.tensor(ring_ns, dtype=torch.long, device=device)
        ring_nd_t = torch.tensor(ring_nd, dtype=torch.long, device=device)
        rand_ns_t = torch.tensor(rand_ns, dtype=torch.long, device=device)
        rand_nd_t = torch.tensor(rand_nd, dtype=torch.long, device=device)

        pos_score  = model.score(z, ps, pd)
        ring_score = model.score(z, ring_ns_t, ring_nd_t)
        rand_score = model.score(z, rand_ns_t, rand_nd_t)

        # GT positive: weight=1.0 / Pseudo positive: weight=pseudo_w
        loss_pos  = F.binary_cross_entropy_with_logits(pos_score,  torch.ones_like(pos_score),  weight=pos_w)
        loss_ring = F.binary_cross_entropy_with_logits(ring_score, torch.zeros_like(ring_score))
        loss_rand = F.binary_cross_entropy_with_logits(rand_score, torch.zeros_like(rand_score))
        loss = loss_pos + (1.0 - rand_prop) * loss_ring + rand_prop * loss_rand

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        opt.step()

        ring_log_f.write(json.dumps({
            "epoch": epoch,
            "thres_outer": round(float(thres_outer), 4),
            "rand_prop": round(float(rand_prop), 4),
            "loss_pos":  round(float(loss_pos.item()), 6),
            "loss_ring": round(float(loss_ring.item()), 6),
            "loss_rand": round(float(loss_rand.item()), 6),
            "ring_detail": epoch_ring_log,
        }) + "\n")

        if epoch % config.VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                zv = model.encode(fd, gd)
                vm = full_val_metrics(model, zv, va, val_pools, retrieved, device)
            mark = ""
            if vm["MRR_end2end"] > best:
                best = vm["MRR_end2end"]; best_epoch = epoch; best_vm = vm
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0; mark = " *"
            else:
                patience += 1
            thres_policy.record(vm["MRR_end2end"])
            print(
                f"epoch={epoch:04d} loss={loss.item():.4f} "
                f"E2E_MRR={vm['MRR_end2end']:.6f} Ret_MRR={vm['MRR_retrieved']:.6f} "
                f"Recall={vm['retrieval_recall']:.4f} H@10_E2E={vm['Hits@10_end2end']:.4f}"
                f"{mark} pat={patience}/{config.PATIENCE} "
                f"thres={thres_outer:.3f} rand_prop={rand_prop:.3f}"
            )
            model.train()
            if patience >= config.PATIENCE:
                break

    ring_log_f.close()

    if best_state is None: raise RuntimeError("checkpoint가 생성되지 않았습니다.")
    ck = out / "checkpoint.pt"; torch.save(best_state, ck)
    pw_str = f"{pseudo_w:.2f}".replace(".", "")
    metrics = {
        "case":             f"case4_curriculum_pw{pw_str}",
        "cutoff":           cutoff,
        "warm_start":       str(warm) if warm else None,
        "best_epoch":       best_epoch,
        "sup_pairs":        len(tr),
        "gt_sup_pairs":     gt_count,
        "pseudo_sup_pairs": pseudo_count,
        "pseudo_w":         pseudo_w,
        "val_pairs":        len(va),
        "neg_per_positive": 2 * M,
        "ring_neg":         M,
        "rand_neg":         M,
        "loss":             "bce_curriculum_weighted_pos",
        "M":                M,
        "SCORE_WINDOW":     SCORE_WINDOW,
        "DELTA":            DELTA,
        "THRES_INNER":      THRES_INNER,
        "MIN_RAND_PROP":    MIN_RAND_PROP,
        "final_thres_outer": round(float(thres_policy.thres), 4),
        "best_val":         best_vm,
        **mp_stats, **info,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("saved:", ck)

    del model, opt, fd, gd, z
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return str(ck)


def main(start=2004, end=2010, pseudo_w=0.5):
    pw_str  = f"{pseudo_w:.2f}".replace(".", "")
    out_dir = EXP / "runs" / f"retrain_case4_curriculum_pw{pw_str}"
    selected = [c for c in FOLDS if start <= c <= end]
    if not selected: raise ValueError("실행할 fold가 없습니다.")
    warm = None
    if selected[0] != 2004:
        prev = FOLDS[FOLDS.index(selected[0]) - 1]
        p = out_dir / f"cutoff_{prev}" / "checkpoint.pt"
        if not p.exists(): raise FileNotFoundError(f"이전 fold checkpoint 없음: {p}")
        warm = str(p)
    for c in selected:
        warm = run_fold(c, pseudo_w, out_dir, warm)
    print(f"\nALL CASE4 CurNM+PW{pseudo_w} FOLDS DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",    type=int,   default=2004, choices=FOLDS)
    ap.add_argument("--end",      type=int,   default=2010, choices=FOLDS)
    ap.add_argument("--pseudo_w", type=float, default=0.5,
                    help="pseudo positive loss weight (0~1, GT=1.0 고정, default: 0.5)")
    a = ap.parse_args()
    main(a.start, a.end, a.pseudo_w)
