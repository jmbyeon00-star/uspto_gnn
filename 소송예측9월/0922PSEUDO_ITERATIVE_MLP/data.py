from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

EDGE_TYPES = [
    ("patent", "cites", "patent"),
    ("patent", "cited_by", "patent"),
    ("patent", "sue", "patent"),
    ("patent", "sued_by", "patent"),
    ("patent", "has_ipc", "ipc"),
    ("ipc", "rev_has_ipc", "patent"),
    ("patent", "owns", "company"),
    ("company", "rev_owns", "patent"),
    ("patent", "prev_owns", "company"),
    ("company", "rev_prev_owns", "patent"),
    ("case", "plaintiff", "company"),
    ("company", "rev_plaintiff", "case"),
    ("case", "defendant", "company"),
    ("company", "rev_defendant", "case"),
    ("case", "plaintiff_patent", "patent"),
    ("patent", "rev_plaintiff_patent", "case"),
    ("case", "defendant_patent", "patent"),
    ("patent", "rev_defendant_patent", "case"),
]

CASE_FORWARD = {
    ("case", "plaintiff", "company"),
    ("case", "defendant", "company"),
    ("case", "plaintiff_patent", "patent"),
    ("case", "defendant_patent", "patent"),
}
CASE_REVERSE = {
    ("company", "rev_plaintiff", "case"),
    ("company", "rev_defendant", "case"),
    ("patent", "rev_plaintiff_patent", "case"),
    ("patent", "rev_defendant_patent", "case"),
}

@dataclass
class StaticFeatureStore:
    patent_x: np.ndarray
    patent_map: Dict[str, int]
    ipc_x: np.ndarray
    ipc_map: Dict[str, int]

    @classmethod
    def load(cls, data_gnn: Path):
        pids = np.load(
            data_gnn / "bge_pca128_feat22_concat_5m_pids.npy",
            allow_pickle=True,
        ).astype(str)
        px = np.load(
            data_gnn / "bge_pca128_feat22_concat_5m.npy"
        ).astype(np.float32)
        pmap = {p:i for i,p in enumerate(pids)}

        iids = np.load(
            data_gnn / "ipc_subclass_ids.npy",
            allow_pickle=True,
        ).astype(str)
        ix = np.load(
            data_gnn / "ipc_node_x.npy"
        ).astype(np.float32)
        ix /= np.linalg.norm(ix, axis=1, keepdims=True) + 1e-8
        imap = {x:i for i,x in enumerate(iids)}
        return cls(px, pmap, ix, imap)

def load_snapshot(snap_dir: Path, year: int):
    candidates = [snap_dir / f"snapshot_{year}_v2.pt", snap_dir / f"snapshot_{year}.pt"]
    for p in candidates:
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"snapshot {year} not found under {snap_dir}")

def build_features(snapshot, static: StaticFeatureStore):
    data = snapshot["data"]
    cutoff = int(snapshot["meta"]["snapshot_year"])

    pat_list = [str(x) for x in snapshot["pat_list"]]
    ipc_list = [str(x) for x in snapshot["ipc_list"]]

    pat = np.zeros((len(pat_list), static.patent_x.shape[1]), dtype=np.float32)
    hit = 0
    for i,p in enumerate(pat_list):
        gi = static.patent_map.get(p)
        if gi is not None:
            pat[i] = static.patent_x[gi]
            hit += 1
    mu = pat.mean(0, keepdims=True)
    sd = pat.std(0, keepdims=True) + 1e-8
    pat = np.clip((pat - mu) / sd, -3.0, 3.0)
    pat /= np.linalg.norm(pat, axis=1, keepdims=True) + 1e-8

    ipc = np.zeros((len(ipc_list), static.ipc_x.shape[1]), dtype=np.float32)
    for i,code in enumerate(ipc_list):
        gi = static.ipc_map.get(code)
        if gi is not None:
            ipc[i] = static.ipc_x[gi]

    company = data["company"].x.cpu().numpy().astype(np.float32)

    case_year = data["case"].year.cpu().numpy().astype(np.float32)
    case_age = (np.clip(cutoff - case_year, 0, 20) / 20.0)[:,None].astype(np.float32)

    feats = {
        "patent": torch.tensor(pat, dtype=torch.float32),
        "ipc": torch.tensor(ipc, dtype=torch.float32),
        "company": torch.tensor(company, dtype=torch.float32),
        "case": torch.tensor(case_age, dtype=torch.float32),
    }
    info = {
        "patent_feature_hit": int(hit),
        "patent_total": int(len(pat_list)),
        "patent_feature_coverage": float(hit / max(len(pat_list), 1)),
    }
    return feats, info

def canonical_sue(snapshot):
    s = snapshot["data"]["patent", "sue", "patent"]
    return s.edge_index.cpu(), s.year.cpu(), s.case_idx.cpu()

def _isin(values: torch.Tensor, selected: torch.Tensor):
    if selected.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return torch.isin(values, selected.cpu())

def split_current_c3(snapshot, cutoff_year: int, sup_ratio: float, val_ratio: float, seed: int):
    _, year, case_idx = canonical_sue(snapshot)
    cases = torch.unique(case_idx[(year == cutoff_year) & (case_idx >= 0)]).cpu()
    if len(cases) < 2:
        raise RuntimeError(f"Too few current C3 cases at {cutoff_year}: {len(cases)}")

    g = torch.Generator().manual_seed(seed)
    perm = cases[torch.randperm(len(cases), generator=g)]
    n_val = max(1, int(round(len(perm) * val_ratio)))
    n_sup = len(perm) - n_val

    return {
        "sup": perm[:n_sup],
        "val": perm[n_sup:],
    }

def sue_edges_for_cases(snapshot, selected_cases: torch.Tensor):
    ei, year, case_idx = canonical_sue(snapshot)
    m = _isin(case_idx, selected_cases)
    return ei[:,m], year[m], case_idx[m]

def build_visible_edge_dict(snapshot, cutoff_year: int, hidden_current_cases: torch.Tensor):
    """
    History C3 (< cutoff): visible MP.
    Current C3 (= cutoff): all supervision/validation -> hidden.

    Also hide Case<->Patent and Case<->Company context for hidden current cases.
    """
    data = snapshot["data"]
    hidden_current_cases = hidden_current_cases.cpu()
    out = {}

    for etype in EDGE_TYPES:
        if etype in {
            ("patent","sue","patent"),
            ("patent","sued_by","patent"),
        }:
            continue
        if etype not in data.edge_types:
            continue

        ei = data[etype].edge_index.cpu()

        if etype in CASE_FORWARD:
            ei = ei[:, ~_isin(ei[0], hidden_current_cases)]
        elif etype in CASE_REVERSE:
            ei = ei[:, ~_isin(ei[1], hidden_current_cases)]

        out[etype] = ei

    sue_ei, sue_year, sue_case = canonical_sue(snapshot)
    history_mask = sue_year < cutoff_year
    hist = sue_ei[:, history_mask]
    out[("patent","sue","patent")] = hist
    out[("patent","sued_by","patent")] = torch.flip(hist, dims=[0])

    return out

def build_homogeneous(snapshot, feats, edge_dict):
    """
    Node order:
      patents -> IPC -> company -> case
    """
    n_pat = feats["patent"].shape[0]
    n_ipc = feats["ipc"].shape[0]
    n_com = feats["company"].shape[0]
    n_case = feats["case"].shape[0]

    offsets = {
        "patent": 0,
        "ipc": n_pat,
        "company": n_pat + n_ipc,
        "case": n_pat + n_ipc + n_com,
    }

    eis = []
    ets = []

    for rel_id, etype in enumerate(EDGE_TYPES):
        ei = edge_dict.get(etype)
        if ei is None or ei.numel() == 0:
            continue
        src_t, _, dst_t = etype
        gei = ei.clone()
        gei[0] += offsets[src_t]
        gei[1] += offsets[dst_t]
        eis.append(gei)
        ets.append(torch.full((gei.shape[1],), rel_id, dtype=torch.long))

    if eis:
        edge_index = torch.cat(eis, dim=1)
        edge_type = torch.cat(ets, dim=0)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "offsets": offsets,
        "counts": {
            "patent": n_pat,
            "ipc": n_ipc,
            "company": n_com,
            "case": n_case,
        }
    }

def build_hard_aux(snapshot, edge_dict):
    """
    Hard candidate = same IPC AND citation-neighbor.
    Patent indices stay local [0, N_pat).
    """
    pat2ipc = {}
    ipc2pat = {}
    pat2cit = {}

    has_ipc = edge_dict.get(("patent","has_ipc","ipc"))
    if has_ipc is not None:
        for p, i in has_ipc.t().tolist():
            pat2ipc.setdefault(int(p), set()).add(int(i))
            ipc2pat.setdefault(int(i), set()).add(int(p))

    for etype in [
        ("patent","cites","patent"),
        ("patent","cited_by","patent"),
    ]:
        ei = edge_dict.get(etype)
        if ei is None:
            continue
        for s,d in ei.t().tolist():
            pat2cit.setdefault(int(s), set()).add(int(d))

    return pat2ipc, ipc2pat, pat2cit

def known_sue_set(snapshot):
    ei, _, _ = canonical_sue(snapshot)
    return set(map(tuple, ei.t().tolist()))

def load_future_c3_targets(snap_dir: Path, base_snapshot, target_years: List[int]):
    future = load_snapshot(snap_dir, max(target_years))
    meta = future["sue_meta"].copy()
    meta = meta[meta["lawsuit_year"].isin(target_years)].copy()

    p2i = {str(k):v for k,v in base_snapshot["pat2idx"].items()}
    pl = meta["patent_plt"].astype(str)
    df = meta["patent_def"].astype(str)

    ok = pl.isin(p2i) & df.isin(p2i)
    eligible = meta[ok].copy()

    src = eligible["patent_plt"].astype(str).map(p2i).to_numpy(np.int64)
    dst = eligible["patent_def"].astype(str).map(p2i).to_numpy(np.int64)

    return src, dst, eligible, {
        "future_c3_all": int(len(meta)),
        "future_c3_eligible": int(len(eligible)),
        "future_c3_coverage": float(len(eligible) / max(len(meta), 1)),
    }
