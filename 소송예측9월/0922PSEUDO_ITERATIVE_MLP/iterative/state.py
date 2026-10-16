from __future__ import annotations
import json
import pickle
from pathlib import Path


class FoldState:
    """
    한 fold 안에서 round가 진행되며 변하는 상태만 저장한다.
    """

    def __init__(self, fold_dir: Path, cutoff: int, ys: int, ye: int):
        self.fold_dir = Path(fold_dir)
        self.cutoff = int(cutoff)
        self.ys = int(ys)
        self.ye = int(ye)

        self.state_dir = self.fold_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.pseudo_path = self.state_dir / "cumulative_pseudo_by_src.pkl"
        self.used_cases_path = self.state_dir / "used_case_ids.pkl"
        self.meta_path = self.state_dir / "state.json"

        self.cumulative_pseudo_by_src = {}
        self.used_case_ids = set()
        self.round = 0
        self.checkpoint = None

        self.load_if_exists()

    def load_if_exists(self):
        if self.pseudo_path.exists():
            with open(self.pseudo_path, "rb") as f:
                x = pickle.load(f)
            self.cumulative_pseudo_by_src = {
                int(s): {int(d) for d in ds}
                for s, ds in x.items()
            }

        if self.used_cases_path.exists():
            with open(self.used_cases_path, "rb") as f:
                self.used_case_ids = set(pickle.load(f))

        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.round = int(meta.get("round", 0))
            self.checkpoint = meta.get("checkpoint")

    def add_pair(self, src: int, dst: int):
        self.cumulative_pseudo_by_src.setdefault(int(src), set()).add(int(dst))

    def mark_case_used(self, case_row_id):
        self.used_case_ids.add(int(case_row_id))

    def save(self):
        with open(self.pseudo_path, "wb") as f:
            pickle.dump(self.cumulative_pseudo_by_src, f)

        with open(self.used_cases_path, "wb") as f:
            pickle.dump(sorted(self.used_case_ids), f)

        self.meta_path.write_text(
            json.dumps(
                {
                    "cutoff": self.cutoff,
                    "target": [self.ys, self.ye],
                    "round": self.round,
                    "checkpoint": self.checkpoint,
                    "n_pseudo_pairs": sum(
                        len(x) for x in self.cumulative_pseudo_by_src.values()
                    ),
                    "n_used_cases": len(self.used_case_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
