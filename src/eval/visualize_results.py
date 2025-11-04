#!/usr/bin/env python3
"""
Visualize evaluation results from CSV files.

Usage:
    python src/eval/visualize_results.py --results eval_results/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed, using matplotlib only")

def load_results(results_dir):
    """Load all result CSV files."""
    results_dir = Path(results_dir)
    results = {}
    
    for csv_file in results_dir.glob("results_*.csv"):
        method_name = csv_file.stem.replace("results_", "")
        results[method_name] = pd.read_csv(csv_file)
    
    return results

def plot_metric_comparison(results, metric_name, save_path=None):
    """Create bar plot comparing a metric across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(results.keys())
    means = [results[m][metric_name].mean() for m in methods]
    stds = [results[m][metric_name].std() for m in methods]
    
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Comparison Across Methods', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_all_metrics(results, output_dir):
    """Create plots for all key metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metrics = ['ADE', 'FDE', 'Nav_Cost', 'Col_Cost', 'Ctrl_Cost', 
               'Traj_Heading', 'Traj_Length', 'Min_Dist', 'Consistency',
               'Avg_Num_Players_Selected']
    
    for metric in metrics:
        plot_metric_comparison(
            results, 
            metric, 
            save_path=output_dir / f"{metric.lower()}_comparison.png"
        )

def plot_metric_heatmap(results, metrics, save_path=None):
    """Create heatmap of normalized metrics across methods."""
    methods = list(results.keys())
    
    # Compute mean for each metric and method
    data = []
    for method in methods:
        row = [results[method][m].mean() for m in metrics]
        data.append(row)
    
    df = pd.DataFrame(data, index=methods, columns=metrics)
    
    # Normalize each column (metric) to [0, 1]
    df_norm = (df - df.min()) / (df.max() - df.min())
    
    # For metrics where lower is better, invert
    lower_is_better = ['ADE', 'FDE', 'Nav_Cost', 'Col_Cost', 'Ctrl_Cost', 
                       'Traj_Heading', 'Traj_Length']
    for metric in lower_is_better:
        if metric in df_norm.columns:
            df_norm[metric] = 1 - df_norm[metric]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(df_norm, annot=True, fmt='.3f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Score (higher is better)'})
    else:
        # Use matplotlib's imshow as fallback
        im = ax.imshow(df_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(len(df_norm.columns)))
        ax.set_yticks(np.arange(len(df_norm.index)))
        ax.set_xticklabels(df_norm.columns, rotation=45, ha='right')
        ax.set_yticklabels(df_norm.index)
        
        # Add text annotations
        for i in range(len(df_norm.index)):
            for j in range(len(df_norm.columns)):
                text = ax.text(j, i, f'{df_norm.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax, label='Normalized Score (higher is better)')
    
    ax.set_title('Normalized Performance Heatmap', fontsize=16)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Methods', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_summary_table(results, output_path=None):
    """Create a formatted summary table."""
    metrics = ['ADE', 'FDE', 'Nav_Cost', 'Col_Cost', 'Ctrl_Cost', 
               'Min_Dist', 'Consistency', 'Avg_Num_Players_Selected']
    
    methods = list(results.keys())
    
    summary_data = []
    for method in methods:
        row = {'Method': method}
        for metric in metrics:
            mean = results[method][metric].mean()
            std = results[method][metric].std()
            row[metric] = f"{mean:.4f} ± {std:.4f}"
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved summary table to {output_path}")
        
        # Also save as pretty-printed text
        txt_path = output_path.replace('.csv', '.txt')
        with open(txt_path, 'w') as f:
            f.write(df.to_string(index=False))
        print(f"Saved pretty-printed table to {txt_path}")
    else:
        print("\n" + "="*100)
        print("SUMMARY TABLE")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--results', type=str, default='eval_results',
                       help='Directory containing result CSV files')
    parser.add_argument('--output', type=str, default='eval_results/plots',
                       help='Output directory for plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively instead of saving')
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    if not results:
        print(f"No result files found in {args.results}")
        return
    
    print(f"Found results for {len(results)} methods: {', '.join(results.keys())}")
    
    # Create summary table
    summary = create_summary_table(
        results, 
        output_path=None if args.show else f"{args.results}/summary_table.csv"
    )
    
    if not args.show:
        # Create plots
        print(f"\nGenerating plots in {args.output}...")
        plot_all_metrics(results, args.output)
        
        # Create heatmap
        metrics = ['ADE', 'FDE', 'Nav_Cost', 'Col_Cost', 'Ctrl_Cost', 
                   'Min_Dist', 'Consistency']
        plot_metric_heatmap(
            results, 
            metrics, 
            save_path=Path(args.output) / "performance_heatmap.png"
        )
        
        print(f"\n{'='*80}")
        print("Visualization complete!")
        print(f"Plots saved to {args.output}/")
        print(f"{'='*80}\n")
    else:
        print("\nShowing plots interactively...")
        for metric in ['ADE', 'FDE', 'Col_Cost', 'Min_Dist', 'Consistency']:
            plot_metric_comparison(results, metric)

if __name__ == "__main__":
    main()

