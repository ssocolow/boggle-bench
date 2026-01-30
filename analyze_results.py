#!/usr/bin/env python3
"""
Analyze Boggle Evaluation Results

Loads model result JSON files from a folder, computes aggregate statistics,
and graphs distributions of numerical results.

Usage:
    python analyze_results.py /path/to/results/folder
    python analyze_results.py /path/to/results/folder --output-dir graphs/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_repeat_results(file_path: Path) -> dict | None:
    """Load a repeat-eval results JSON file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        # Validate it's a repeat-eval file
        if "runs" in data and isinstance(data["runs"], list):
            return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {file_path}: {e}", file=sys.stderr)
    return None


def load_single_result(file_path: Path) -> dict | None:
    """Load a single model result JSON file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        # Validate it's a single result file (has wordsFound but no runs)
        if "wordsFound" in data and "runs" not in data:
            return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {file_path}: {e}", file=sys.stderr)
    return None


def extract_metrics_from_run(run: dict) -> dict | None:
    """Extract numerical metrics from a single run."""
    if "error" in run:
        return None

    return {
        "wordScore": run.get("wordScore", 0),
        "wordsFound": len(run.get("wordsFound", [])),
        "mistakenWords": len(run.get("mistakenWords", [])),
        "transcriptionErrors": run.get("transcriptionErrors", 0),
        "totalWordsAttempted": run.get("totalWordsAttempted", 0),
    }


def extract_metrics_from_single(result: dict, correct_grid: list = None) -> dict:
    """Extract numerical metrics from a single result file."""
    metrics = {
        "wordScore": result.get("wordScore", 0),
        "wordsFound": len(result.get("wordsFound", [])),
        "mistakenWords": len(result.get("mistakenWords", [])),
    }

    # Calculate transcription errors if we have the grid
    if correct_grid and "transcriptionGrid" in result:
        trans_grid = result["transcriptionGrid"]
        errors = 25 - sum(
            1 for i in range(5) for j in range(5)
            if trans_grid[i][j] == correct_grid[i][j]
        )
        metrics["transcriptionErrors"] = errors

    return metrics


def compute_statistics(values: list) -> dict:
    """Compute aggregate statistics for a list of values."""
    if not values:
        return {}

    arr = np.array(values)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def plot_distribution(values: list, title: str, xlabel: str, output_path: Path = None):
    """Plot a histogram distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    arr = np.array(values)

    # Determine bin count
    n_bins = min(30, max(10, len(set(values))))

    ax.hist(arr, bins=n_bins, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(arr), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(arr):.2f}')
    ax.axvline(np.median(arr), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(arr):.2f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_all_distributions(metrics_by_name: dict, model_name: str, output_dir: Path = None):
    """Plot distributions for all metrics."""
    metric_labels = {
        "wordScore": "Word Score",
        "wordsFound": "Valid Words Found",
        "mistakenWords": "Mistaken Words",
        "transcriptionErrors": "Transcription Errors",
        "totalWordsAttempted": "Total Words Attempted",
    }

    for metric_name, values in metrics_by_name.items():
        if not values:
            continue

        label = metric_labels.get(metric_name, metric_name)
        title = f"{model_name}: {label} Distribution (n={len(values)})"

        output_path = None
        if output_dir:
            safe_metric = metric_name.replace(" ", "_").lower()
            output_path = output_dir / f"{safe_metric}_distribution.png"

        plot_distribution(values, title, label, output_path)


def plot_combined_boxplot(metrics_by_name: dict, model_name: str, output_path: Path = None):
    """Create a combined boxplot for all metrics (normalized)."""
    fig, axes = plt.subplots(1, len(metrics_by_name), figsize=(4 * len(metrics_by_name), 6))

    if len(metrics_by_name) == 1:
        axes = [axes]

    metric_labels = {
        "wordScore": "Word Score",
        "wordsFound": "Valid Words",
        "mistakenWords": "Mistaken Words",
        "transcriptionErrors": "Trans. Errors",
        "totalWordsAttempted": "Total Attempted",
    }

    for ax, (metric_name, values) in zip(axes, metrics_by_name.items()):
        if not values:
            continue

        label = metric_labels.get(metric_name, metric_name)
        bp = ax.boxplot(values, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel(label)
        ax.set_xticklabels([])

        # Add mean marker
        mean_val = np.mean(values)
        ax.scatter([1], [mean_val], color='red', marker='D', s=50, zorder=3, label=f'Mean: {mean_val:.1f}')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f"{model_name}: Metric Distributions", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_repeat_results(data: dict, output_dir: Path = None):
    """Analyze results from a repeat-eval file."""
    model_name = data.get("model", "Unknown Model")
    runs = data.get("runs", [])

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Total runs: {data.get('totalRuns', len(runs))}")
    print(f"Successful runs: {data.get('successfulRuns', sum(1 for r in runs if 'error' not in r))}")
    print(f"{'='*60}")

    # Collect metrics from all runs
    metrics_by_name = defaultdict(list)

    for run in runs:
        metrics = extract_metrics_from_run(run)
        if metrics:
            for name, value in metrics.items():
                metrics_by_name[name].append(value)

    # Compute and print statistics
    print("\nAggregate Statistics:")
    print("-" * 60)

    stats_output = {}
    for metric_name, values in metrics_by_name.items():
        stats = compute_statistics(values)
        stats_output[metric_name] = stats

        print(f"\n{metric_name}:")
        print(f"  Count:  {stats['count']}")
        print(f"  Mean:   {stats['mean']:.2f}")
        print(f"  Std:    {stats['std']:.2f}")
        print(f"  Min:    {stats['min']:.0f}")
        print(f"  Max:    {stats['max']:.0f}")
        print(f"  Median: {stats['median']:.0f}")
        print(f"  Q25:    {stats['q25']:.0f}")
        print(f"  Q75:    {stats['q75']:.0f}")

    # Plot distributions
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving graphs to {output_dir}...")

        plot_all_distributions(metrics_by_name, model_name, output_dir)
        plot_combined_boxplot(metrics_by_name, model_name, output_dir / "combined_boxplot.png")

        # Save statistics to JSON
        stats_path = output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump({
                "model": model_name,
                "totalRuns": data.get("totalRuns"),
                "successfulRuns": data.get("successfulRuns"),
                "statistics": stats_output
            }, f, indent=2)
        print(f"  Saved: {stats_path}")
    else:
        print("\nDisplaying graphs (use --output-dir to save)...")
        plot_all_distributions(metrics_by_name, model_name)
        plot_combined_boxplot(metrics_by_name, model_name)

    return stats_output


def analyze_folder(folder_path: Path, output_dir: Path = None, correct_grid: list = None):
    """Analyze all result files in a folder."""
    json_files = list(folder_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {folder_path}", file=sys.stderr)
        return

    print(f"Found {len(json_files)} JSON files in {folder_path}")

    # Try to load as repeat-eval files first
    repeat_results = []
    single_results = []

    for file_path in json_files:
        if file_path.name == "index.json":
            continue

        data = load_repeat_results(file_path)
        if data:
            repeat_results.append((file_path, data))
            continue

        data = load_single_result(file_path)
        if data:
            single_results.append((file_path, data))

    # Analyze repeat-eval files
    for file_path, data in repeat_results:
        file_output_dir = None
        if output_dir:
            safe_name = file_path.stem.replace(" ", "_")
            file_output_dir = output_dir / safe_name
        analyze_repeat_results(data, file_output_dir)

    # If we have multiple single results, aggregate them
    if single_results and not repeat_results:
        print(f"\nAnalyzing {len(single_results)} single result files...")

        metrics_by_name = defaultdict(list)
        model_names = set()

        for file_path, result in single_results:
            model_names.add(result.get("model", "Unknown"))
            metrics = extract_metrics_from_single(result, correct_grid)
            for name, value in metrics.items():
                metrics_by_name[name].append(value)

        model_name = ", ".join(sorted(model_names)) if len(model_names) <= 3 else f"{len(model_names)} models"

        print(f"\n{'='*60}")
        print(f"Models: {model_name}")
        print(f"Total files: {len(single_results)}")
        print(f"{'='*60}")

        # Compute and print statistics
        print("\nAggregate Statistics:")
        print("-" * 60)

        for metric_name, values in metrics_by_name.items():
            stats = compute_statistics(values)
            print(f"\n{metric_name}:")
            print(f"  Count:  {stats['count']}")
            print(f"  Mean:   {stats['mean']:.2f}")
            print(f"  Std:    {stats['std']:.2f}")
            print(f"  Min:    {stats['min']:.0f}")
            print(f"  Max:    {stats['max']:.0f}")
            print(f"  Median: {stats['median']:.0f}")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_all_distributions(metrics_by_name, model_name, output_dir)
            plot_combined_boxplot(metrics_by_name, model_name, output_dir / "combined_boxplot.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze Boggle evaluation results")
    parser.add_argument("input", help="Input file (repeat-eval JSON) or folder containing result files")
    parser.add_argument("--output-dir", "-o", help="Output directory for graphs and statistics")
    parser.add_argument("--correct-grid", help="Correct grid string for calculating transcription errors in single files")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    correct_grid = None
    if args.correct_grid:
        rows = args.correct_grid.split(";")
        correct_grid = [row.split(",") for row in rows]

    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        # Single file analysis
        data = load_repeat_results(input_path)
        if data:
            analyze_repeat_results(data, output_dir)
        else:
            print(f"Error: {input_path} is not a valid repeat-eval results file", file=sys.stderr)
            sys.exit(1)
    else:
        # Folder analysis
        analyze_folder(input_path, output_dir, correct_grid)


if __name__ == "__main__":
    main()
