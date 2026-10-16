from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import torch

M = 20
MAX_HIST_PROP = 0.6
PI_INIT = 1.0
PI_MIN = 0.1
SHRINKAGE = 0.03
MIN_RAND_PROP = 0.3
CACHE_THRESHOLD = 0.5
STD_WEIGHT = 0.006
HISTORY_LEN = 5


class AdaptivePi:
    def __init__(self):
        self.pi = PI_INIT
        self.prev_ap = None

    def record(self, ap):
        old = self.pi
        if self.prev_ap is None:
            self.prev_ap = float(ap)
            return old, self.pi
        if ap > self.prev_ap:
            self.pi = max(PI_MIN, self.pi - SHRINKAGE)
        else:
            self.pi = min(1.0, self.pi + SHRINKAGE)
        self.prev_ap = float(ap)
        return old, self.pi


class CurNM:
    def __init__(self, pools, pos_by_src, res_by_src, n_pat, rng):
        self.pools = pools
        self.pos_by_src = pos_by_src
        self.res_by_src = res_by_src
        self.n_pat = int(n_pat)
        self.rng = rng
        self.policy = AdaptivePi()
        self.cache = {}
        self.score_hist = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self.cache_started = False
        self.cache_start_epoch = None

    def blocked(self, s):
        return {int(s)} | self.pos_by_src.get(int(s), set()) | self.res_by_src.get(int(s), set())

    def _global_random(self, s, n, extra=None):
        blocked = self.blocked(s)
        if extra:
            blocked |= set(map(int, extra))
        out = []
        while len(out) < n:
            x = int(self.rng.integers(0, self.n_pat))
            if x not in blocked and x not in out:
                out.append(x)
        return out

    def _structural(self, s, n, extra=None):
        blocked = self.blocked(s)
        if extra:
            blocked |= set(map(int, extra))
        hp = self.pools.get(int(s), self.pools.get(str(int(s)), []))
        pool = [int(x) for x in hp if int(x) not in blocked]
        self.rng.shuffle(pool)
        out = pool[:n]
        if len(out) < n:
            out.extend(self._global_random(s, n-len(out), extra=set(out) | blocked))
        return out

    def fresh_candidates(self, s, n=M):
        num_hist = int(M * MAX_HIST_PROP)
        nh = num_hist if n == M else min(max(num_hist // 2, 1), n)
        nr = n - nh
        hist = self._structural(s, nh)
        rand = self._global_random(s, nr, extra=hist)
        return hist + rand

    def candidates(self, s):
        s = int(s)
        if not self.cache_started or s not in self.cache:
            return self.fresh_candidates(s, M)

        cached = list(self.cache[s])[:M // 2]
        fresh = self.fresh_candidates(s, M // 2)
        used = set(cached)
        clean_fresh = []

        for x in fresh:
            if x not in used:
                clean_fresh.append(x)
                used.add(x)

        while len(clean_fresh) < M // 2:
            x = self._global_random(s, 1, extra=used)[0]
            clean_fresh.append(x)
            used.add(x)

        return cached + clean_fresh

    def build_candidate_pairs(self, train_pairs):
        src, dst = [], []
        for s, _ in train_pairs:
            cand = self.candidates(s)
            src.extend([int(s)] * len(cand))
            dst.extend(cand)
        return src, dst

    def select(self, scores):
        num_cand = int(scores.numel())
        n_sel = max(1, int(round(self.policy.pi * num_cand)))
        n_sel = min(n_sel, num_cand)
        idx = torch.argsort(scores.detach(), descending=True)[:n_sel]
        rand_prop = max(n_sel / num_cand, MIN_RAND_PROP)
        return idx, n_sel, rand_prop

    def maybe_start_cache(self, epoch):
        if not self.cache_started and self.policy.pi <= CACHE_THRESHOLD:
            self.cache_started = True
            self.cache_start_epoch = int(epoch)

    def update_cache(self, epoch, cand_src, cand_dst, cand_scores):
        if not self.cache_started:
            return

        grouped = defaultdict(list)
        scores = cand_scores.detach().cpu().numpy()

        for s, d, score in zip(cand_src, cand_dst, scores):
            key = (int(s), int(d))
            self.score_hist[key].append(float(score))
            hist = list(self.score_hist[key])
            std = float(np.std(hist)) if len(hist) > 1 else 0.0
            progress = min(1.0, epoch / 1000.0)
            update_score = float(score) - STD_WEIGHT * progress * std
            grouped[int(s)].append((int(d), update_score))

        for s, items in grouped.items():
            if len(items) <= M // 2:
                self.cache[s] = [d for d, _ in items]
                continue
            vals = np.asarray([v for _, v in items], dtype=float)
            vals -= vals.max()
            probs = np.exp(vals)
            probs /= probs.sum()
            idx = self.rng.choice(len(items), size=M // 2, replace=False, p=probs)
            self.cache[s] = [items[int(i)][0] for i in idx]

    def sample_external_random(self, train_pairs):
        src, dst = [], []
        for s, _ in train_pairs:
            neg = self._global_random(s, M)
            src.extend([int(s)] * M)
            dst.extend(neg)
        return src, dst
