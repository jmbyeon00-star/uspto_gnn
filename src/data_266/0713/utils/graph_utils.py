"""utils/graph_utils.py — Graph filtering and subgraph helpers."""
from __future__ import annotations


def filter_nodes(nodes: list, types: set | None = None) -> list:
    if types is None:
        return nodes
    return [n for n in nodes if n["type"] in types]


def filter_edges(edges: list, threshold: float = 0.0,
                 relations: set | None = None) -> list:
    result = []
    for e in edges:
        if relations and e["relation"] not in relations:
            continue
        if e.get("importance", 1.0) < threshold and e["relation"] not in ("owner", "ipc"):
            continue
        result.append(e)
    return result


def nodes_in_hop(nodes: list, max_hop: int) -> list:
    """Return nodes whose min_hop <= max_hop (None = always included)."""
    return [n for n in nodes if n.get("min_hop") is None or n["min_hop"] <= max_hop]


def get_path_nodes(graph: dict, ep: dict) -> set:
    return set(ep.get("path", []))


def get_path_edge_pairs(ep: dict) -> set:
    return {(e["source"], e["target"]) for e in ep.get("edges", [])}


def node_by_id(nodes: list) -> dict:
    return {n["id"]: n for n in nodes}


def edge_set_for_nodes(edges: list, node_ids: set) -> list:
    return [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]


def top_k_edges(edges: list, key: str = "importance", k: int = 10) -> set:
    sorted_e = sorted(edges, key=lambda e: -(e.get(key) or 0))
    return {e["id"] for e in sorted_e[:k]}
