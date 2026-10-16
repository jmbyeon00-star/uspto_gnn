from __future__ import annotations
import pickle
from pathlib import Path
import pandas as pd

import config
from data import load_snapshot


def norm(x):
    if pd.isna(x):
        return None
    s = str(x)
    return s[:-2] if s.endswith(".0") else s


def relation_maps(obj):
    d = obj["data"]
    p2ipc, ipc2p, adj = {}, {}, {}

    if ("patent", "has_ipc", "ipc") in d.edge_types:
        for p, i in d["patent", "has_ipc", "ipc"].edge_index.cpu().t().tolist():
            p2ipc.setdefault(int(p), set()).add(int(i))
            ipc2p.setdefault(int(i), set()).add(int(p))

    for et in [
        ("patent", "cites", "patent"),
        ("patent", "cited_by", "patent"),
    ]:
        if et in d.edge_types:
            for s, t in d[et].edge_index.cpu().t().tolist():
                adj.setdefault(int(s), set()).add(int(t))

    return p2ipc, ipc2p, adj


def khop(adj, src, k=3):
    seen = {int(src)}
    frontier = {int(src)}
    layers = []

    for _ in range(k):
        nxt = set()
        for x in frontier:
            nxt |= adj.get(x, set())
        nxt -= seen
        layers.append(nxt)
        seen |= nxt
        frontier = nxt

        if not frontier:
            layers += [set()] * (k - len(layers))
            break

    return layers


def ownership_groups():
    oc = pd.read_parquet(config.OWNERSHIP_PATH).copy()

    ren = {}
    if "patent" in oc.columns and "patent_id" not in oc.columns:
        ren["patent"] = "patent_id"
    if "assignee" in oc.columns and "assignee_id" not in oc.columns:
        ren["assignee"] = "assignee_id"

    oc = oc.rename(columns=ren)
    oc["patent_id"] = oc["patent_id"].map(norm)
    oc["assignee_id"] = oc["assignee_id"].map(norm)
    oc["start_dt"] = pd.to_datetime(oc["start_dt"], errors="coerce")
    oc["end_dt"] = pd.to_datetime(oc["end_dt"], errors="coerce")

    return {str(a): g for a, g in oc.groupby("assignee_id", sort=False)}


def portfolio(groups, aid, cutoff_dt, p2i):
    aid = norm(aid)
    if aid is None or aid not in groups:
        return set()

    q = groups[aid]
    q = q[
        (q.start_dt.isna() | q.start_dt.le(cutoff_dt))
        & (q.end_dt.isna() | q.end_dt.gt(cutoff_dt))
    ]

    return {
        p2i[str(p)]
        for p in q.patent_id.dropna().astype(str).unique()
        if str(p) in p2i
    }


class UnifiedPoolManager:
    """
    Fold별 unified base hard-negative pool 관리자.

    - Case1 + Case3 src는 이미 fixed_data/unified_base_pools에 존재.
    - Case2에서 신규 src가 채택되면 해당 src만 lazy-build.
    - base pool 자체에서 pseudo를 삭제하지 않는다.
    - 실제 negative = base - real positive - cumulative pseudo positive.
    """

    def __init__(self, root: Path, cutoff: int):
        self.root = Path(root)
        self.cutoff = int(cutoff)
        self.path = (
            self.root
            / "fixed_data"
            / "unified_base_pools"
            / f"pool_{self.cutoff}.pkl"
        )

        with open(self.path, "rb") as f:
            self.obj = pickle.load(f)

        self.obj["pools"] = {
            int(s): {int(x) for x in pool}
            for s, pool in self.obj["pools"].items()
        }
        self.obj["real_positive_by_src"] = {
            int(s): {int(x) for x in ds}
            for s, ds in self.obj["real_positive_by_src"].items()
        }

        self._base_snapshot = None
        self._ownership = None
        self._maps = None

    def has_src(self, src: int) -> bool:
        return int(src) in self.obj["pools"]

    def _prepare_lazy_context(self):
        if self._base_snapshot is None:
            self._base_snapshot = load_snapshot(config.SNAP_DIR, self.cutoff)
        if self._ownership is None:
            self._ownership = ownership_groups()
        if self._maps is None:
            self._maps = relation_maps(self._base_snapshot)

    def build_single_src_pool(self, src: int, candidate_assignee_id):
        self._prepare_lazy_context()

        src = int(src)
        base = self._base_snapshot
        p2i = {str(k): int(v) for k, v in base["pat2idx"].items()}
        p2ipc, ipc2p, adj = self._maps

        same = set()
        for ip in p2ipc.get(src, set()):
            same |= ipc2p.get(ip, set())

        layers = khop(adj, src, 3)
        c1 = layers[0] if len(layers) > 0 else set()
        c2 = c1 | (layers[1] if len(layers) > 1 else set())
        c3 = c2 | (layers[2] if len(layers) > 2 else set())

        cutoff_dt = pd.Timestamp(f"{self.cutoff}-12-30")
        pf = portfolio(
            self._ownership,
            candidate_assignee_id,
            cutoff_dt,
            p2i,
        )

        real = self.obj["real_positive_by_src"].get(src, set())
        forbid = {src} | set(real)

        A = (same & c1) - forbid
        B = (same & c2) - forbid - A
        C = (same & c3) - forbid - A - B
        D = (pf & (same | c3)) - forbid - A - B - C

        return A | B | C | D

    def ensure_src(self, src: int, candidate_assignee_id=None):
        src = int(src)

        if self.has_src(src):
            return False, len(self.obj["pools"][src])

        if candidate_assignee_id is None:
            raise ValueError(
                f"src={src} is absent from unified pool; "
                "candidate_assignee_id is required for lazy-build."
            )

        new_pool = self.build_single_src_pool(src, candidate_assignee_id)
        self.obj["pools"][src] = set(new_pool)

        self.obj.setdefault("source_meta", {})[src] = {
            "from_c3": False,
            "from_case1": False,
            "from_case2_lazy": True,
            "n_unified": len(new_pool),
        }

        return True, len(new_pool)

    def available_negatives(self, src: int, cumulative_pseudo_by_src):
        src = int(src)

        base = set(self.obj["pools"].get(src, set()))
        real = set(self.obj["real_positive_by_src"].get(src, set()))
        pseudo = set(cumulative_pseudo_by_src.get(src, set()))

        neg = base - real - pseudo

        assert src not in neg
        assert not (neg & real)
        assert not (neg & pseudo)

        return sorted(neg)

    def save(self):
        payload = dict(self.obj)
        payload["pools"] = {
            int(s): sorted(pool)
            for s, pool in self.obj["pools"].items()
        }
        payload["real_positive_by_src"] = {
            int(s): sorted(ds)
            for s, ds in self.obj["real_positive_by_src"].items()
        }

        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        tmp.replace(self.path)

    def verify(self, cumulative_pseudo_by_src=None):
        cumulative_pseudo_by_src = cumulative_pseudo_by_src or {}

        for src in self.obj["pools"]:
            neg = set(self.available_negatives(src, cumulative_pseudo_by_src))
            real = self.obj["real_positive_by_src"].get(src, set())
            pseudo = set(cumulative_pseudo_by_src.get(src, set()))

            assert not (neg & set(real))
            assert not (neg & pseudo)
            assert src not in neg

        return True
