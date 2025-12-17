"""
graph_utils.py

Tiny helpers for visualizing causal-learn graphs (GeneralGraph) using Graphviz (DOT).

Requires:
  pip install graphviz
And to render PNG/SVG you also need the Graphviz system package installed
(e.g., apt-get install graphviz / brew install graphviz).

Core functions:
  - generalgraph_to_dot(g, title=None): returns DOT string
  - draw_generalgraph(g, title=None): returns graphviz.Source (displayable in notebooks)
"""

from __future__ import annotations

from typing import Optional

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Endpoint import Endpoint


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _endpoint_to_arrow(ep: Endpoint, *, left: bool) -> str:
    """
    Map causal-learn endpoints to DOT arrow shapes.
    We treat:
      - TAIL   => no arrow (none)
      - ARROW  => normal arrow
      - CIRCLE => open circle (odot)
    """
    name = getattr(ep, "name", str(ep))
    if name == "ARROW":
        return "normal"
    if name == "CIRCLE":
        return "odot"
    # TAIL or unknown => none
    return "none"


def generalgraph_to_dot(g: GeneralGraph, title: Optional[str] = None) -> str:
    """
    Convert a causallearn GeneralGraph into a DOT graph string.

    Handles edge endpoint types (TAIL/ARROW/CIRCLE) by mapping them to
    Graphviz 'arrowtail'/'arrowhead' shapes.
    """
    nodes = [n.get_name() for n in g.get_nodes()]

    lines = []
    lines.append("digraph G {")
    lines.append('  graph [rankdir=LR, bgcolor="white"];')
    lines.append('  node  [shape=box, style="rounded,filled", fillcolor="white", color="black", fontname="Helvetica"];')
    lines.append('  edge  [color="black", penwidth=1.2, fontname="Helvetica"];')

    if title:
        lines.append(f'  labelloc="t"; label="{_escape(title)}"; fontsize=16;')

    # Nodes
    for n in nodes:
        lines.append(f'  "{_escape(n)}";')

    # Edges
    for e in g.get_graph_edges():
        u = e.get_node1().get_name()
        v = e.get_node2().get_name()
        ep_u = e.get_endpoint1()
        ep_v = e.get_endpoint2()

        arrowtail = _endpoint_to_arrow(ep_u, left=True)
        arrowhead = _endpoint_to_arrow(ep_v, left=False)

        # Use dir=both to show arrowtail + arrowhead simultaneously
        # If both are "none", Graphviz will still draw a line; that's ok.
        lines.append(
            f'  "{_escape(u)}" -> "{_escape(v)}" '
            f'[dir=both, arrowtail={arrowtail}, arrowhead={arrowhead}];'
        )

    lines.append("}")
    return "\n".join(lines)


def draw_generalgraph(g: GeneralGraph, title: Optional[str] = None):
    """
    Return a graphviz.Source object for notebook display or rendering.

    Example:
      from graph_utils import draw_generalgraph
      src = draw_generalgraph(g, "PCMCI+ aggregate")
      src.render("out/pcmci_graph", format="png", cleanup=True)
    """
    try:
        from graphviz import Source
    except Exception as e:
        raise ImportError(
            "graphviz (python package) is required. Install with: pip install graphviz"
        ) from e

    dot = generalgraph_to_dot(g, title=title)
    return Source(dot)
