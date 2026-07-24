"""model.py — SimpleCatHeteroGAT (phase3 실험 2: SimpleConcat + Margin Loss)

Architecture:
  Patent input : cat(struct 3d, FAISS 1024d) → 1027d → Linear(1027, 128) → LeakyReLU
  Company input: Linear(1024d FAISS avg, 128d)
  IPC input    : Embedding(n_ipc, 128d)

  L1: GATConv(128, 16, heads=8, attn_drop=0.3) → BN → LeakyReLU
  L2: GATConv(128, 32, heads=4)                → BN → LeakyReLU
  L3: GATConv(128, 32, heads=4)                → BN → Tanh
  L4: GATConv(128, 32, heads=2)                → (raw output, 64d)

  HID=128, OUT=64
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

STRUCT_DIM = 3
FAISS_DIM  = 1024
DIM_TOTAL  = STRUCT_DIM + FAISS_DIM  # 1027
HIDDEN_DIM = 128
OUT_DIM    = 64

EDGE_TYPES = [
    ("patent",  "cites",        "patent"),
    ("patent",  "cited_by",     "patent"),
    ("patent",  "has_ipc",      "ipc"),
    ("ipc",     "rev_has_ipc",  "patent"),
    ("company", "owns",         "patent"),
    ("patent",  "owned_by",     "company"),
    ("patent",  "sued",         "patent"),
    ("patent",  "sued_by",      "patent"),
]

_CONV_ACTS = [F.leaky_relu, F.leaky_relu, torch.tanh]  # L1, L2, L3


class SimpleLinearProj(nn.Module):
    """cat(struct, FAISS) → Linear(1027→hid) → LeakyReLU"""
    def __init__(self, dim_total=DIM_TOTAL, hid=HIDDEN_DIM, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(dim_total, hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.leaky_relu(self.proj(x)))


class SimpleCatHeteroGAT(nn.Module):
    """phase3 실험 2: SimpleConcat + 4-layer HeteroGAT + BatchNorm"""
    def __init__(self, n_ipc, hid=HIDDEN_DIM, out=OUT_DIM, drop=0.2):
        super().__init__()
        self.hid = hid
        self.out = out

        self.pat_proj = SimpleLinearProj(DIM_TOTAL, hid, dropout=drop)
        self.co_proj  = nn.Linear(FAISS_DIM, hid)
        self.ipc_emb  = nn.Embedding(n_ipc, hid)

        layer_cfg = [
            (hid // 8, 8, 0.3),  # L1: attn_drop=0.3
            (hid // 4, 4, 0.0),  # L2
            (hid // 4, 4, 0.0),  # L3
            (out // 2, 2, 0.0),  # L4
        ]
        self.convs = nn.ModuleList([
            HeteroConv({
                et: GATConv(hid, head_dim, heads=n_heads,
                            dropout=attn_drop,
                            concat=True, add_self_loops=False)
                for et in EDGE_TYPES
            }, aggr="mean")
            for head_dim, n_heads, attn_drop in layer_cfg
        ])

        # BatchNorm for L1, L2, L3 only
        self.bns = nn.ModuleList([
            nn.ModuleDict({nt: nn.BatchNorm1d(hid)
                           for nt in ["patent", "company", "ipc"]})
            for _ in range(3)
        ])
        self.drop = nn.Dropout(drop)

    def forward(self, hetero_data):
        x_pat = torch.cat([hetero_data["patent"].struct,
                            hetero_data["patent"].x], dim=1)
        x = {
            "patent":  self.pat_proj(x_pat),
            "company": F.leaky_relu(self.co_proj(hetero_data["company"].x)),
            "ipc":     self.ipc_emb.weight,
        }
        for conv, bn, act in zip(self.convs[:3], self.bns, _CONV_ACTS):
            x_new = conv(x, hetero_data.edge_index_dict)
            x = {
                k: self.drop(act(bn[k](v.view(-1, self.hid))))
                if k in bn else self.drop(act(v))
                for k, v in x_new.items()
            }
        x_out = self.convs[3](x, hetero_data.edge_index_dict)
        return {k: v.view(-1, self.out) for k, v in x_out.items()}
