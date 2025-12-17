"""
PCMCI+ helper for *mixed* data using Tigramite's RegressionCI.

This is meant to complement the provided algorithms.py without modifying it.
All graphs returned are causallearn.graph.GeneralGraph to keep one common format.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite.independence_tests.regressionCI import RegressionCI


def pcmciplus_mixed(
    datasets: Union[Dict[object, pd.DataFrame], pd.DataFrame],
    *,
    tau_max: int = 1,
    sig_level: float = 0.05,
    col_types: Optional[Union[pd.Series, Sequence[int]]] = None,
    verbosity: int = 0,
) -> GeneralGraph:
    """
    Run PCMCI+ with RegressionCI for mixed data.

    Parameters
    ----------
    datasets : pd.DataFrame or Dict[object, pd.DataFrame]
        Either:
          - dict: {dataset_key: dataframe(T_i x N)} (recommended for multiple machines)
          - single dataframe(T x N) for one dataset
        Columns must match across datasets.

    tau_max : int
        Maximum time lag (includes contemporaneous links with tau_min=0).

    sig_level : float
        pc_alpha in Tigramite (threshold for PC step). You may also apply FDR later.

    col_types : pd.Series or Sequence[int], optional
        Column types, 0=continuous, 1=discrete/categorical. If a Series, its index must
        match columns. Required for RegressionCI.

    verbosity : int
        Verbosity level for PCMCI (0=silent, 1=progress, 2+=debug).

    Returns
    -------
    GeneralGraph
        causallearn GeneralGraph (summary graph collapsed across lags):
        Adds i->j if Tigramite finds any lagged or contemporaneous link i -> j.
        
    Raises
    ------
    ValueError
        If col_types is not provided (required for RegressionCI).
    """
    # Validate col_types is provided
    if col_types is None:
        raise ValueError("col_types must be provided for RegressionCI (0=continuous, 1=discrete).")

    # Normalize input: convert single DataFrame to dict
    if isinstance(datasets, pd.DataFrame):
        dfs = {0: datasets.copy()}
        analysis_mode = "single"
    else:
        dfs = {k: v.copy() for k, v in datasets.items()}
        analysis_mode = "multiple"

    # Get column order from first dataframe
    cols = list(next(iter(dfs.values())).columns)
    
    # Validate we have data
    if len(cols) == 0:
        raise ValueError("Input dataframe(s) have no columns.")

    # Convert col_types to numpy array
    if isinstance(col_types, pd.Series):
        col_types_arr = col_types.reindex(cols).values.astype(int)
    else:
        col_types_arr = np.asarray(list(col_types), dtype=int)
    
    # Validate col_types length matches columns
    if len(col_types_arr) != len(cols):
        raise ValueError(
            f"col_types length ({len(col_types_arr)}) must match number of columns ({len(cols)})"
        )

    # Build Tigramite data dict in the required format
    data_dict = {}
    dtype_dict = {}

    for k, df in dfs.items():
        # Ensure column order matches
        df = df[cols]
        X = df.values.astype(float)
        data_dict[k] = X
        # data_type must have same shape as the dataset array (T_i, N)
        dtype_dict[k] = np.tile(col_types_arr.reshape(1, -1), (X.shape[0], 1))

    # Create Tigramite DataFrame
    if analysis_mode == "single":
        dataframe = pp.DataFrame(
            data=data_dict[0],
            data_type=dtype_dict[0],
            datatime=None,
            var_names=cols,
            analysis_mode=analysis_mode,
        )
    else:
        dataframe = pp.DataFrame(
            data=data_dict,
            data_type=dtype_dict,
            datatime=None,
            var_names=cols,
            analysis_mode=analysis_mode,
        )

    # Run PCMCI+ with RegressionCI
    cond_ind_test = RegressionCI()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)
    out = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=sig_level)

    # Collapse tigramite's (i,j,tau) graph into a single summary graph
    graph_arr = out["graph"]  # shape (N, N, tau_max+1)
    
    directed = set()
    bidirected = set()
    undirected = set()

    def _norm(s) -> str:
        """Normalize edge string (handle None and whitespace)."""
        if s is None:
            return ""
        return str(s).strip()

    # Parse PCMCI+ output graph
    for i in range(len(cols)):
        for j in range(len(cols)):
            for tau in range(0, tau_max + 1):
                cell = _norm(graph_arr[i, j, tau])
                if cell == "":
                    continue
                    
                # Tigramite docs: graph[i,j,tau] = '-->' denotes i -> j at lag tau
                if cell == "-->":
                    directed.add((cols[i], cols[j]))
                elif cell == "<--":
                    directed.add((cols[j], cols[i]))
                elif cell == "<->":
                    # Bidirected edge (latent confounder)
                    if cols[i] != cols[j]:
                        bidirected.add(tuple(sorted((cols[i], cols[j]))))
                elif cell == "---":
                    # Undirected edge (not oriented)
                    if cols[i] != cols[j]:
                        undirected.add(tuple(sorted((cols[i], cols[j]))))
                else:
                    # Best-effort handling for other edge types (e.g., o->, <-o, +->)
                    if cell.endswith("->"):
                        directed.add((cols[i], cols[j]))
                    elif cell.startswith("<-"):
                        directed.add((cols[j], cols[i]))

    # Build GeneralGraph
    node_map = {c: GraphNode(c) for c in cols}
    g = GeneralGraph(list(node_map.values()))

    # Add directed edges (skip self-loops)
    for src, dst in sorted(directed):
        if src == dst:
            continue
        g.add_edge(Edge(node_map[src], node_map[dst], Endpoint.TAIL, Endpoint.ARROW))

    # Add bidirected edges
    for a, b in sorted(bidirected):
        g.add_edge(Edge(node_map[a], node_map[b], Endpoint.ARROW, Endpoint.ARROW))

    # Add undirected edges
    for a, b in sorted(undirected):
        g.add_edge(Edge(node_map[a], node_map[b], Endpoint.TAIL, Endpoint.TAIL))

    return g
