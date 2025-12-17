"""
Serialization + simple aggregation utilities for causallearn.graph.GeneralGraph.

We store graphs as JSON so that:
- outputs are human-readable
- artifacts are easy to version / diff
- you can reload graphs later to draw / evaluate

Graph format:
{
  "nodes": ["A", "B", ...],
  "edges": [
      {"u": "A", "v": "B", "ep_u": "TAIL", "ep_v": "ARROW"}
  ],
  "metadata": {...}   # optional
}
"""

from __future__ import annotations

from dataclasses import dataclass, asdict as _dataclass_asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
from pathlib import Path

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def _ep_to_str(ep: Endpoint) -> str:
    """Convert Endpoint enum to string."""
    # Endpoint is an enum-like class in causal-learn
    try:
        return ep.name
    except Exception:
        return str(ep)


def _str_to_ep(s: str) -> Endpoint:
    """Convert string to Endpoint enum."""
    # Accept "TAIL" as well as "Endpoint.TAIL"
    s = s.strip()
    if s.startswith("Endpoint."):
        s = s.split(".", 1)[1]
    if hasattr(Endpoint, s):
        return getattr(Endpoint, s)
    raise ValueError(f"Unknown Endpoint string: {s!r}")


def generalgraph_to_dict(g: GeneralGraph, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert a GeneralGraph to a JSON-serializable dictionary."""
    nodes = [n.get_name() for n in g.get_nodes()]
    edges_out: List[Dict[str, str]] = []
    for e in g.get_graph_edges():
        n1 = e.get_node1().get_name()
        n2 = e.get_node2().get_name()
        ep1 = _ep_to_str(e.get_endpoint1())
        ep2 = _ep_to_str(e.get_endpoint2())
        edges_out.append({"u": n1, "v": n2, "ep_u": ep1, "ep_v": ep2})
    out: Dict[str, Any] = {"nodes": nodes, "edges": edges_out}
    if metadata is not None:
        out["metadata"] = metadata
    return out


def dict_to_generalgraph(d: Dict[str, Any]) -> GeneralGraph:
    """Convert a dictionary back to a GeneralGraph."""
    nodes = [GraphNode(n) for n in d["nodes"]]
    node_map = {n.get_name(): n for n in nodes}
    g = GeneralGraph(nodes)
    for ed in d.get("edges", []):
        u = node_map[ed["u"]]
        v = node_map[ed["v"]]
        ep_u = _str_to_ep(ed["ep_u"])
        ep_v = _str_to_ep(ed["ep_v"])
        g.add_edge(Edge(u, v, ep_u, ep_v))
    return g


def save_graph_json(path: str | Path, g: GeneralGraph, *, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save a GeneralGraph to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = generalgraph_to_dict(g, metadata=metadata)
    path.write_text(json.dumps(d, indent=2))


def load_graph_json(path: str | Path) -> Tuple[GeneralGraph, Dict[str, Any]]:
    """Load a GeneralGraph from a JSON file."""
    path = Path(path)
    d = json.loads(path.read_text())
    g = dict_to_generalgraph(d)
    return g, d.get("metadata", {})


def _edge_key(e: Edge) -> Tuple[str, str, str, str]:
    """Create a hashable key for an edge."""
    return (
        e.get_node1().get_name(),
        e.get_node2().get_name(),
        _ep_to_str(e.get_endpoint1()),
        _ep_to_str(e.get_endpoint2()),
    )


def _edge_key_to_str(key: Tuple[str, str, str, str]) -> str:
    """Convert edge key tuple to a JSON-serializable string."""
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"


def _str_to_edge_key(s: str) -> Tuple[str, str, str, str]:
    """Convert string back to edge key tuple."""
    parts = s.split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid edge key string: {s!r}")
    return (parts[0], parts[1], parts[2], parts[3])


@dataclass
class AggregateInfo:
    """Information about graph aggregation results."""
    n_graphs: int
    min_support: int
    edge_support: Dict[Tuple[str, str, str, str], int]
    
    def to_json_dict(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.
        
        Fixed: Tuple keys in edge_support are converted to strings for JSON compatibility.
        """
        return {
            "n_graphs": self.n_graphs,
            "min_support": self.min_support,
            "edge_support": {
                _edge_key_to_str(k): v for k, v in self.edge_support.items()
            }
        }
    
    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "AggregateInfo":
        """Reconstruct AggregateInfo from a JSON dictionary."""
        edge_support = {
            _str_to_edge_key(k): v for k, v in d.get("edge_support", {}).items()
        }
        return cls(
            n_graphs=d["n_graphs"],
            min_support=d["min_support"],
            edge_support=edge_support,
        )


def aggregate_info_asdict(info: AggregateInfo) -> Dict[str, Any]:
    """
    Safely convert AggregateInfo to a JSON-serializable dictionary.
    
    Use this instead of dataclasses.asdict() to handle tuple keys properly.
    """
    return info.to_json_dict()


def aggregate_graphs(
    graphs: Iterable[GeneralGraph],
    *,
    min_support: int = 1,
) -> Tuple[GeneralGraph, AggregateInfo]:
    """
    Simple edge-support aggregation: keep an edge if it appears (same endpoints) in >= min_support graphs.

    Works even when different graphs have different node sets (we build the union of nodes).
    
    Parameters
    ----------
    graphs : Iterable[GeneralGraph]
        Collection of graphs to aggregate
    min_support : int
        Minimum number of graphs an edge must appear in to be included (default: 1)
        
    Returns
    -------
    Tuple[GeneralGraph, AggregateInfo]
        The aggregated graph and information about the aggregation
    """
    graphs = list(graphs)
    if len(graphs) == 0:
        raise ValueError("aggregate_graphs() received 0 graphs.")

    # Union of node names across graphs
    node_names = sorted({n.get_name() for g in graphs for n in g.get_nodes()})
    nodes = [GraphNode(n) for n in node_names]
    node_map = {n.get_name(): n for n in nodes}
    out = GeneralGraph(nodes)

    support: Dict[Tuple[str, str, str, str], int] = {}
    for g in graphs:
        seen = set()
        for e in g.get_graph_edges():
            k = _edge_key(e)
            if k in seen:
                continue
            seen.add(k)
            support[k] = support.get(k, 0) + 1

    for (u, v, ep_u, ep_v), c in support.items():
        if c >= min_support:
            if u in node_map and v in node_map:
                out.add_edge(Edge(node_map[u], node_map[v], _str_to_ep(ep_u), _str_to_ep(ep_v)))

    return out, AggregateInfo(n_graphs=len(graphs), min_support=min_support, edge_support=support)
