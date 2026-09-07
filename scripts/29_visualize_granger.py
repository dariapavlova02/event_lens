#!/usr/bin/env python3
"""
Visualizes Granger Causality Results for Thesis
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

# Data from previous run
data = [
    {'feature': 'Graph Density', 'p_value': 0.0010, 'lag': 5, 'type': 'Structural'},
    {'feature': 'Modularity', 'p_value': 0.0027, 'lag': 2, 'type': 'Structural'},
    {'feature': 'Degree Gini', 'p_value': 0.0313, 'lag': 5, 'type': 'Structural'},
    {'feature': 'Stability', 'p_value': 0.0421, 'lag': 3, 'type': 'Structural'},
    {'feature': 'Volume (nodes)', 'p_value': 0.0657, 'lag': 5, 'type': 'Baseline'},
    {'feature': 'Edges', 'p_value': 0.1295, 'lag': 2, 'type': 'Baseline'},
    {'feature': 'Assortativity', 'p_value': 0.2389, 'lag': 3, 'type': 'Structural'},
    {'feature': 'Transitivity', 'p_value': 0.3525, 'lag': 1, 'type': 'Structural'},
]

df = pd.DataFrame(data).sort_values('p_value')

plt.figure(figsize=(10, 6))
colors = ['firebrick' if t == 'Structural' and p < 0.05 else 'gray' for t, p in zip(df['type'], df['p_value'])]

bars = plt.barh(df['feature'], -np.log10(df['p_value']), color=colors)

# Add p=0.05 threshold line
plt.axvline(-np.log10(0.05), color='black', linestyle='--', label='p=0.05 Significance')

plt.xlabel('-log10(p-value) [Higher is Better]')
plt.title('Predictive Power of Graph Features for Volatility (Granger Causality)')
plt.legend()
plt.tight_layout()

output_file = RESULTS_DIR / 'granger_causality_graph.png'
plt.savefig(output_file, dpi=300)
print(f"Saved to {output_file}")
