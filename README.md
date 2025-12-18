# Causal Discovery for Predictive Maintenance

This project implements and evaluates multiple causal discovery algorithms on the Microsoft Azure Predictive Maintenance dataset to identify causal relationships between sensor telemetry, errors, maintenance events, and equipment failures.

## Prerequisites

### Required Python Packages

```bash
pip install pandas numpy networkx scikit-learn matplotlib tigramite lingam
```

### Dataset

1. Download the Microsoft Azure Predictive Maintenance dataset from:
   https://github.com/Azure/predictive-maintenance/tree/master/data

2. Download these 5 CSV files:
   - `PdM_telemetry.csv`
   - `PdM_errors.csv`
   - `PdM_failures.csv`
   - `PdM_maint.csv`
   - `PdM_machines.csv`

3. Place them in a `./data/` directory:
   ```
   Code/
   ├── data/
   │   ├── PdM_telemetry.csv
   │   ├── PdM_errors.csv
   │   ├── PdM_failures.csv
   │   ├── PdM_maint.csv
   │   └── PdM_machines.csv
   ```

## Running the Code

### Option 1: Run Analysis Scripts

**Edge Stability Analysis** - Analyzes consistency of discovered edges across machines:

```bash
python edge_stability_eval.py
```

This computes edge stability metrics and generates visualizations in `outputs/edge_stability/`.

**Graph Aggregation & Evaluation** - Aggregates graphs and evaluates against ground truth:

```bash
python aggregate_and_eval.py
```

This aggregates graphs across models, evaluates telemetry→failure relationships, and displays statistics.

### Option 2: Interactive Jupyter Notebook

```bash
jupyter notebook eval.ipynb
```

Then run the cells to:
- Load and aggregate graphs
- Evaluate against ground truth telemetry→failure edges
- Visualize results with heatmaps
- Compare specific edges across algorithms

## Basic Usage in Python

```python
# Import functions
from aggregate_and_eval import (
    aggregate_graphs_by_algorithm,
    evaluate_telemetry_failure_edges,
    plot_telemetry_failure_heatmap,
    get_edge_comparison
)

# Aggregate graphs (keep edges appearing in ≥30% of machines)
agg = aggregate_graphs_by_algorithm(min_frequency=0.3)

# Evaluate against known telemetry→failure relationships
eval_df = evaluate_telemetry_failure_edges(agg)

# Plot results
plot_telemetry_failure_heatmap(eval_df)

# Compare specific edge across algorithms
get_edge_comparison(agg, 'rotate', 'fail_comp2')
```

## Output Files

All results are saved in `outputs/`:

- **`outputs/edge_stability/`**: Stability analysis results
  - `stability_summary.csv` - Overall statistics per algorithm
  - `*_edge_details.csv` - Per-algorithm edge frequencies
  - `*.png` - Visualization plots

- **`outputs/model=modelX/`**: Raw algorithm results (pre-computed)
  - JSON files for each algorithm and model combination

- **`outputs/aggregated_graphs_pruned.json`**: Aggregated consensus graphs

## Key Functions

### `aggregate_graphs_by_algorithm(min_frequency=0.3)`
Aggregates graphs across all models for each algorithm and prunes edges below the frequency threshold.

### `evaluate_telemetry_failure_edges(agg)`
Evaluates which algorithms discovered the known telemetry→failure relationships:
- volt → fail_comp1
- rotate → fail_comp2
- pressure → fail_comp3
- vibration → fail_comp4

### `plot_telemetry_failure_heatmap(eval_df)`
Creates a heatmap visualization showing which algorithms found which edges.

### `get_edge_comparison(agg, source, target)`
Shows how each algorithm handled a specific edge.

## Troubleshooting

**Import errors**: Install missing packages with `pip install package_name`

**Data not found**: Ensure CSV files are in `./data/` directory

**Jupyter won't start**: Install with `pip install jupyter notebook`

## File Descriptions

- **`aggregate_and_eval.py`**: Main evaluation module for aggregation and ground truth comparison
- **`edge_stability_eval.py`**: Edge stability analysis and visualization
- **`eval.ipynb`**: Interactive Jupyter notebook for exploration
- **`NOTEBOOK_USAGE.md`**: Detailed usage documentation
- **`QUICK_REFERENCE.txt`**: Quick command reference

## Dataset Information

Microsoft Azure Predictive Maintenance Dataset contains:
- Telemetry: Sensor readings (voltage, rotation, pressure, vibration)
- Errors: Error codes from machines
- Failures: Component failure events
- Maintenance: Maintenance records
- Machines: Machine metadata (model, age)

## Algorithms Evaluated

The project includes results from 7 causal discovery algorithms:
- PCMCI+ (PC-MCI Plus)
- VARLiNGAM (Vector Autoregressive Linear Non-Gaussian Acyclic Model)
- GCMVL (Granger Causality with MV Learning)
- CBNB (Constraint-Based Network with Bootstrapping) - window and event variants
- NBCB (Network Building with Constraint-Based) - window and event variants

Results are aggregated across 4 machine models (model1, model2, model3, model4).
