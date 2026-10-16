import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

def key_of(etype):
    return "__".join(etype)

class MLPDecoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, src, dst):
        z_src = z[src]
        z_dst = z[dst]

        x = torch.cat([z_src, z_dst], dim=-1)

        return self.mlp(x).squeeze(-1)


class StreamingHeteroLayer(nn.Module):
    """
    Relation-specific SAGEConv with streaming sum aggregation.
    Semantically matches HeteroConv(..., aggr='sum') but avoids torch.stack
    over all relation outputs, which was causing CUDA OOM on later snapshots.
    """
    def __init__(self, edge_types, hidden):
        super().__init__()
        self.edge_types = list(edge_types)
        self.convs = nn.ModuleDict({
            key_of(e): SAGEConv((hidden, hidden), hidden)
            for e in self.edge_types
        })

    def forward(self, h, edge_dict):
        accum = {k: torch.zeros_like(v) for k,v in h.items()}
        touched = {k: False for k in h}

        for etype in self.edge_types:
            ei = edge_dict.get(etype)
            if ei is None or ei.numel() == 0:
                continue
            src_t, _, dst_t = etype
            out = self.convs[key_of(etype)](
                (h[src_t], h[dst_t]),
                ei,
            )
            accum[dst_t] = accum[dst_t] + out
            touched[dst_t] = True

        return {
            k: (accum[k] if touched[k] else torch.zeros_like(v))
            for k,v in h.items()
        }

class Model(nn.Module):
    def __init__(self, input_dims, hidden, edge_types, dropout, **kwargs):
        super().__init__()
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden)
            for ntype,dim in input_dims.items()
        })
        self.layer1 = StreamingHeteroLayer(edge_types, hidden)
        self.layer2 = StreamingHeteroLayer(edge_types, hidden)
        self.norm1 = nn.ModuleDict({k: nn.LayerNorm(hidden) for k in input_dims})
        self.norm2 = nn.ModuleDict({k: nn.LayerNorm(hidden) for k in input_dims})
        self.dropout = nn.Dropout(dropout)
        self.decoder = MLPDecoder(hidden)

    def encode(self, feats, graph):
        h = {
            k: F.relu(self.proj[k](v))
            for k,v in feats.items()
        }
        edge_dict = graph["edge_dict"]

        o1 = self.layer1(h, edge_dict)
        h1 = {}
        for k in h:
            v = self.norm1[k](o1[k] + h[k])
            h1[k] = self.dropout(F.relu(v))

        o2 = self.layer2(h1, edge_dict)
        h2 = {}
        for k in h1:
            h2[k] = self.norm2[k](o2[k] + h1[k])

        return h2["patent"]

    def score(self, z_patent, src, dst):
        return self.decoder(z_patent, src, dst)
