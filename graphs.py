"""Unified graph helpers.

- Graph representation: causallearn.graph.GeneralGraph (used everywhere)
- Serialization + aggregation: graph_io.py
- Visualization: graph_utils.py (optional)

Import from here to avoid confusion:
    from graphs_fixed import save_graph_json, load_graph_json, aggregate_graphs
"""

from graph_io import (
    generalgraph_to_dict,
    dict_to_generalgraph,
    save_graph_json,
    load_graph_json,
    aggregate_graphs,
    AggregateInfo,
)

try:
    from graph_utils import generalgraph_to_dot, draw_generalgraph
except ImportError:
    def generalgraph_to_dot(g, **kwargs):
        raise NotImplementedError("graph_utils.py not available. Install or create it for visualization.")
    
    def draw_generalgraph(g, **kwargs):
        raise NotImplementedError("graph_utils.py not available. Install or create it for visualization.")
