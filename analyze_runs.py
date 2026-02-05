#!/usr/bin/env python3
"""Analyze wordScore statistics across multiple runs for each model."""

import argparse
import json
import os
import statistics
from collections import defaultdict

DATA_DIR = "data/game1"
RUNS = ["run1", "run2", "run3", "run4", "run5"]


def load_scores():
    """Load wordScores for each model from all runs."""
    model_scores = defaultdict(list)

    for run in RUNS:
        run_dir = os.path.join(DATA_DIR, run)
        if not os.path.exists(run_dir):
            print(f"Warning: {run_dir} does not exist")
            continue

        for filename in os.listdir(run_dir):
            if filename == "index.json" or not filename.endswith(".json"):
                continue

            filepath = os.path.join(run_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            model_name = data.get("model", filename.replace(".json", ""))
            word_score = data.get("wordScore", 0)
            model_scores[model_name].append(word_score)

    return model_scores


def calculate_statistics(model_scores):
    """Calculate mean and standard deviation for each model."""
    stats = []

    for model, scores in model_scores.items():
        n = len(scores)
        mean = statistics.mean(scores) if n > 0 else 0
        stdev = statistics.stdev(scores) if n > 1 else 0

        stats.append({
            "model": model,
            "n": n,
            "scores": scores,
            "mean": mean,
            "stdev": stdev,
        })

    # Sort by mean score descending
    stats.sort(key=lambda x: x["mean"], reverse=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze wordScore statistics across multiple runs")
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path for stats (e.g., data/game1/stats.json)"
    )
    args = parser.parse_args()

    model_scores = load_scores()
    stats = calculate_statistics(model_scores)

    print("=" * 70)
    print(f"{'Model':<30} {'N':>4} {'Mean':>10} {'Std Dev':>10} {'Scores'}")
    print("=" * 70)

    for s in stats:
        scores_str = ", ".join(str(x) for x in s["scores"])
        print(f"{s['model']:<30} {s['n']:>4} {s['mean']:>10.2f} {s['stdev']:>10.2f}   [{scores_str}]")

    print("=" * 70)

    if args.output:
        # Prepare output data (exclude raw scores to keep file smaller)
        output_data = {
            "models": [
                {
                    "model": s["model"],
                    "mean": round(s["mean"], 2),
                    "stdev": round(s["stdev"], 2),
                    "n": s["n"],
                }
                for s in stats
            ]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nStats written to {args.output}")


if __name__ == "__main__":
    main()
