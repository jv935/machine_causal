"""
Graph Aggregation and Evaluation Functions

Compatible with existing graph format:
{
    'nodes': [...],
    'edges': [{'u': source, 'v': target, 'ep_u': 'TAIL', 'ep_v': 'ARROW'}, ...],
    'metadata': {...}
}
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def aggregate_graphs_by_algorithm(
    outputs_dir: str = "outputs",
    mixed_only: bool = True,
    min_frequency: float = 0.0
) -> Dict[str, Dict]:
    """
    Aggregate graphs across all models for each algorithm.
    Output format is compatible with existing graphing functions.

    Args:
        outputs_dir: Directory containing model outputs
        mixed_only: If True, only process algorithms ending with '_mixed'
        min_frequency: Minimum frequency threshold to include edge (0.0 to 1.0)
                      Edges appearing in fewer machines will be pruned

    Returns:
        Dictionary mapping algorithm_name to:
        {
            'nodes': list of all nodes,
            'edges': list of edge dicts with 'u', 'v', 'ep_u', 'ep_v', 'frequency', 'count',
            'metadata': {
                'total_machines': int,
                'n_edges_before_pruning': int,
                'n_edges_after_pruning': int,
                'min_frequency': float
            }
        }

    Example:
        >>> agg = aggregate_graphs_by_algorithm(min_frequency=0.3)
        >>> agg['pcmci_mixed']['edges'][0]
        {'u': 'rotate', 'v': 'volt', 'ep_u': 'TAIL', 'ep_v': 'ARROW',
         'frequency': 0.438, 'count': 32}
    """
    outputs_path = Path(outputs_dir)

    # Collect all algorithm names
    algorithm_names = set()
    for model_dir in outputs_path.glob("model=*"):
        for result_file in model_dir.glob("*.json"):
            if result_file.name in ["meta.json", "machine_selection.json"]:
                continue
            algorithm_names.add(result_file.stem)

    # Filter to mixed only if requested
    if mixed_only:
        algorithm_names = {name for name in algorithm_names if name.endswith("_mixed")}

    print(f"Aggregating {len(algorithm_names)} algorithms (min_frequency={min_frequency:.1%})...")

    aggregated = {}

    for algorithm_name in sorted(algorithm_names):
        edge_counts = defaultdict(int)
        edge_endpoints = {}  # Store endpoint info for each edge
        all_nodes = set()
        total_machines = 0

        # Load from all models
        for model_dir in outputs_path.glob("model=*"):
            result_file = model_dir / f"{algorithm_name}.json"
            if not result_file.exists():
                continue

            with open(result_file, 'r') as f:
                data = json.load(f)

            per_machine = data.get('per_machine', {})

            for machine_id, graph_data in per_machine.items():
                if graph_data is None:
                    continue

                total_machines += 1

                # Collect nodes
                nodes = graph_data.get('nodes', [])
                all_nodes.update(nodes)

                # Count edges
                edges = graph_data.get('edges', [])
                for edge in edges:
                    u = edge['u']
                    v = edge['v']
                    ep_u = edge.get('ep_u', 'TAIL')
                    ep_v = edge.get('ep_v', 'ARROW')

                    # Simple edge key (u, v)
                    edge_key = (u, v)
                    edge_counts[edge_key] += 1

                    # Store endpoint info (use most common)
                    if edge_key not in edge_endpoints:
                        edge_endpoints[edge_key] = {'ep_u': ep_u, 'ep_v': ep_v}

        # Create edge list with frequencies
        edges_before_pruning = []
        for (u, v), count in edge_counts.items():
            frequency = count / total_machines if total_machines > 0 else 0
            ep_info = edge_endpoints.get((u, v), {'ep_u': 'TAIL', 'ep_v': 'ARROW'})

            edges_before_pruning.append({
                'u': u,
                'v': v,
                'ep_u': ep_info['ep_u'],
                'ep_v': ep_info['ep_v'],
                'frequency': frequency,
                'count': count,
                'total_machines': total_machines
            })

        # Prune edges below threshold
        edges_after_pruning = [e for e in edges_before_pruning if e['frequency'] >= min_frequency]

        # Sort by frequency (highest first)
        edges_after_pruning.sort(key=lambda x: x['frequency'], reverse=True)

        aggregated[algorithm_name] = {
            'nodes': sorted(all_nodes),
            'edges': edges_after_pruning,
            'metadata': {
                'total_machines': total_machines,
                'n_edges_before_pruning': len(edges_before_pruning),
                'n_edges_after_pruning': len(edges_after_pruning),
                'min_frequency': min_frequency,
                'algorithm': algorithm_name
            }
        }

        print(f"  {algorithm_name}: {len(edges_after_pruning)} edges "
              f"({len(edges_before_pruning)} before pruning), {total_machines} machines")

    return aggregated


def prune_graph(graph_data: Dict, min_frequency: float) -> Dict:
    """
    Prune edges from an already aggregated graph based on frequency threshold.

    Args:
        graph_data: Graph dict from aggregate_graphs_by_algorithm()
        min_frequency: New minimum frequency threshold

    Returns:
        New graph dict with pruned edges

    Example:
        >>> agg = aggregate_graphs_by_algorithm(min_frequency=0.0)
        >>> pruned = prune_graph(agg['pcmci_mixed'], min_frequency=0.5)
    """
    edges = graph_data['edges']
    pruned_edges = [e for e in edges if e['frequency'] >= min_frequency]

    return {
        'nodes': graph_data['nodes'],
        'edges': pruned_edges,
        'metadata': {
            **graph_data['metadata'],
            'n_edges_after_pruning': len(pruned_edges),
            'min_frequency': min_frequency
        }
    }


def evaluate_telemetry_failure_edges(
    aggregated_graphs: Dict[str, Dict],
    ground_truth_edges: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Evaluate which algorithms discovered the known telemetry→failure relationships.

    Args:
        aggregated_graphs: Output from aggregate_graphs_by_algorithm()
        ground_truth_edges: List of (source, target) tuples.
                           Default: [('volt', 'fail_comp1'), ('rotate', 'fail_comp2'),
                                    ('pressure', 'fail_comp3'), ('vibration', 'fail_comp4')]

    Returns:
        DataFrame with columns:
            - algorithm: Algorithm name
            - edge: The telemetry→failure edge (as string)
            - source: Source node
            - target: Target node
            - discovered: Whether edge was found (True/False)
            - frequency: Edge frequency if found, 0.0 otherwise
            - count: Number of machines with edge

    Example:
        >>> agg = aggregate_graphs_by_algorithm()
        >>> eval_df = evaluate_telemetry_failure_edges(agg)
        >>> print(eval_df[eval_df['discovered']])
    """
    # Default ground truth
    if ground_truth_edges is None:
        ground_truth_edges = [
            ('volt', 'fail_comp1'),
            ('rotate', 'fail_comp2'),
            ('pressure', 'fail_comp3'),
            ('vibration', 'fail_comp4')
        ]

    print(f"\nEvaluating {len(ground_truth_edges)} ground truth telemetry→failure edges:")
    for src, tgt in ground_truth_edges:
        print(f"  {src} → {tgt}")

    results = []

    for algorithm_name, graph_data in aggregated_graphs.items():
        edges = graph_data['edges']
        total_machines = graph_data['metadata']['total_machines']

        for src, tgt in ground_truth_edges:
            # Search for this edge
            edge_found = None
            for edge in edges:
                if edge['u'] == src and edge['v'] == tgt:
                    edge_found = edge
                    break

            if edge_found:
                results.append({
                    'algorithm': algorithm_name,
                    'edge': f"{src}→{tgt}",
                    'source': src,
                    'target': tgt,
                    'discovered': True,
                    'frequency': edge_found['frequency'],
                    'count': edge_found['count'],
                    'total_machines': total_machines
                })
            else:
                results.append({
                    'algorithm': algorithm_name,
                    'edge': f"{src}→{tgt}",
                    'source': src,
                    'target': tgt,
                    'discovered': False,
                    'frequency': 0.0,
                    'count': 0,
                    'total_machines': total_machines
                })

    df = pd.DataFrame(results)

    # Print summary
    summary = df.groupby('algorithm').agg({
        'discovered': 'sum',
        'frequency': lambda x: x[x > 0].mean() if any(x > 0) else 0
    }).reset_index()
    summary.columns = ['algorithm', 'n_discovered', 'avg_frequency']
    summary['total_edges'] = len(ground_truth_edges)
    summary['recall'] = summary['n_discovered'] / len(ground_truth_edges)
    summary = summary.sort_values('recall', ascending=False)

    print("\n" + "="*70)
    print("SUMMARY: Telemetry→Failure Edge Discovery")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70 + "\n")

    return df


def plot_telemetry_failure_heatmap(
    evaluation_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    output_path: Optional[str] = None
):
    """
    Plot heatmap showing frequency of telemetry→failure edges by algorithm.

    Args:
        evaluation_df: Output from evaluate_telemetry_failure_edges()
        figsize: Figure size
        output_path: Optional path to save figure

    Returns:
        matplotlib figure and axis
    """
    import matplotlib.pyplot as plt

    # Pivot to create matrix
    pivot = evaluation_df.pivot(index='algorithm', columns='edge', values='frequency')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(pivot.values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)

    # Add annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if val > 0:
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=11, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                       color='gray', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Frequency', fontsize=11)

    # Labels
    ax.set_xlabel('Telemetry → Failure Edge', fontsize=12, fontweight='bold')
    ax.set_ylabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_title('Discovery of Known Telemetry→Failure Relationships\n'
                'Frequency across all machines',
                fontsize=13, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")

    return fig, ax


def save_aggregated_graphs(
    aggregated_graphs: Dict[str, Dict],
    output_path: str = "outputs/aggregated_graphs.json"
):
    """
    Save aggregated graphs to JSON file.

    Args:
        aggregated_graphs: Output from aggregate_graphs_by_algorithm()
        output_path: Path to save JSON file

    Example:
        >>> agg = aggregate_graphs_by_algorithm()
        >>> save_aggregated_graphs(agg)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    output_data = {}
    for algo_name, data in aggregated_graphs.items():
        output_data[algo_name] = {
            'nodes': data['nodes'],
            'edges': data['edges'],
            'metadata': data['metadata']
        }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved aggregated graphs to {output_path}")


def get_edge_comparison(
    aggregated_graphs: Dict[str, Dict],
    source: str,
    target: str
) -> pd.DataFrame:
    """
    Compare how different algorithms handled a specific edge.

    Args:
        aggregated_graphs: Output from aggregate_graphs_by_algorithm()
        source: Source node
        target: Target node

    Returns:
        DataFrame showing frequency across algorithms

    Example:
        >>> agg = aggregate_graphs_by_algorithm()
        >>> get_edge_comparison(agg, 'rotate', 'fail_comp2')
    """
    results = []

    for algorithm_name, graph_data in aggregated_graphs.items():
        edges = graph_data['edges']
        total_machines = graph_data['metadata']['total_machines']

        # Search for edge
        edge_found = None
        for edge in edges:
            if edge['u'] == source and edge['v'] == target:
                edge_found = edge
                break

        if edge_found:
            results.append({
                'algorithm': algorithm_name,
                'frequency': edge_found['frequency'],
                'count': edge_found['count'],
                'total_machines': total_machines,
                'discovered': True
            })
        else:
            results.append({
                'algorithm': algorithm_name,
                'frequency': 0.0,
                'count': 0,
                'total_machines': total_machines,
                'discovered': False
            })

    df = pd.DataFrame(results)
    df = df.sort_values('frequency', ascending=False)

    return df


def print_summary_statistics(aggregated_graphs: Dict[str, Dict]):
    """
    Print summary statistics for aggregated graphs.

    Args:
        aggregated_graphs: Output from aggregate_graphs_by_algorithm()
    """
    print("\n" + "="*70)
    print("AGGREGATED GRAPHS SUMMARY")
    print("="*70)

    summary_data = []
    for algo_name, data in aggregated_graphs.items():
        edges = data['edges']
        metadata = data['metadata']

        if edges:
            frequencies = [e['frequency'] for e in edges]
            high_freq = sum(1 for f in frequencies if f >= 0.5)
            med_freq = sum(1 for f in frequencies if 0.3 <= f < 0.5)
            low_freq = sum(1 for f in frequencies if f < 0.3)
            avg_freq = np.mean(frequencies)
            max_freq = max(frequencies)
        else:
            high_freq = med_freq = low_freq = 0
            avg_freq = max_freq = 0.0

        summary_data.append({
            'Algorithm': algo_name,
            'Total Machines': metadata['total_machines'],
            'Edges (after prune)': len(edges),
            'Edges (before prune)': metadata['n_edges_before_pruning'],
            'Avg Freq': f"{avg_freq:.3f}",
            'Max Freq': f"{max_freq:.3f}",
            'High (≥50%)': high_freq,
            'Med (30-50%)': med_freq,
            'Low (<30%)': low_freq
        })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("Testing aggregation functions...\n")

    # Test 1: Aggregate with no pruning
    print("TEST 1: Aggregate all edges (no pruning)")
    print("-" * 70)
    agg = aggregate_graphs_by_algorithm(min_frequency=0.0)
    print_summary_statistics(agg)

    # Test 2: Aggregate with pruning
    print("\nTEST 2: Aggregate with 30% frequency threshold")
    print("-" * 70)
    agg_pruned = aggregate_graphs_by_algorithm(min_frequency=0.3)
    print_summary_statistics(agg_pruned)

    # Test 3: Evaluate telemetry→failure edges
    print("\nTEST 3: Evaluate telemetry→failure edges")
    print("-" * 70)
    eval_df = evaluate_telemetry_failure_edges(agg)

    print("\nDetailed results:")
    print(eval_df[eval_df['discovered']].to_string(index=False))

    # Test 4: Compare specific edge
    print("\n\nTEST 4: Compare 'rotate → fail_comp2' across algorithms")
    print("-" * 70)
    comparison = get_edge_comparison(agg, 'rotate', 'fail_comp2')
    print(comparison.to_string(index=False))

    # Test 5: Save to file
    print("\n\nTEST 5: Saving aggregated graphs")
    print("-" * 70)
    save_aggregated_graphs(agg_pruned, "outputs/aggregated_graphs_pruned.json")

    print("\n✓ All tests completed!")
