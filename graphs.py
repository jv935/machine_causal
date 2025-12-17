"""Unified graph helpers.

- Graph representation: causallearn.graph.GeneralGraph (used everywhere)
- Serialization + aggregation: graph_io.py
- Visualization: graph_utils.py (optional)

Import from here to avoid confusion:
    from graphs_fixed import save_graph_json, load_graph_json, aggregate_graphs
"""

# Fixed: Import from actual file name (graph_io.py, not graph_io_fixed.py)
from graph_io import (
    generalgraph_to_dict,
    dict_to_generalgraph,
    save_graph_json,
    load_graph_json,
    aggregate_graphs,
    AggregateInfo,
)

# Fixed: Make graph_utils optional since it wasn't provided
try:
    from graph_utils import generalgraph_to_dot, draw_generalgraph
except ImportError:
    # graph_utils.py not available - provide stub functions
    def generalgraph_to_dot(g, **kwargs):
        raise NotImplementedError("graph_utils.py not available. Install or create it for visualization.")
    
    def draw_generalgraph(g, **kwargs):
        raise NotImplementedError("graph_utils.py not available. Install or create it for visualization.")
