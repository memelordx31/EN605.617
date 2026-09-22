#!/usr/bin/env python3
import sys
import csv
import os
import matplotlib.pyplot as plt


def parse_statistics_csv(path):
    with open(path, newline="") as f:
        raw_lines = [line.rstrip("\n") for line in f]
    blank_idx = next((i for i, l in enumerate(raw_lines) if l.strip() == ""), len(raw_lines))
    block1 = raw_lines[:blank_idx]
    block2 = raw_lines[blank_idx + 1:]

    iterations, gpu_times, cpu_times = [], [], []
    reader1 = csv.DictReader(block1)
    for row in reader1:
        iterations.append(int(row["iteration"]))
        gpu_times.append(float(row["gpu_ns"]))
        cpu_times.append(float(row["cpu_ns"]))

    stats = {}
    block2 = [l for l in block2 if l.strip() != ""]
    if block2:
        reader2 = csv.DictReader(block2)
        for row in reader2:
            stats[row["stat"]] = {
                "gpu_ns": float(row["gpu_ns"]),
                "cpu_ns": float(row["cpu_ns"]),
            }

    return iterations, gpu_times, cpu_times, stats


def ns_to_ms(values):
    return [v / 1e6 for v in values]


def plot_timeline(iterations, gpu_times, cpu_times, out_dir):
    gpu_ms = ns_to_ms(gpu_times)
    cpu_ms = ns_to_ms(cpu_times)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, gpu_ms, marker="o", label="GPU")
    plt.plot(iterations, cpu_ms, marker="o", label="CPU")
    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title("GPU vs CPU Execution Time per Iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "gpu_vs_cpu_timeline.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_speedup(iterations, gpu_times, cpu_times, out_dir):
    speedup = [c / g if g > 0 else 0 for g, c in zip(gpu_times, cpu_times)]

    plt.figure(figsize=(8, 5))
    plt.bar(iterations, speedup, color="mediumseagreen")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Speedup (CPU time / GPU time)")
    plt.title("GPU Speedup over CPU per Iteration")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "speedup.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_summary_stats(stats, out_dir):
    if not stats:
        print("No summary-stat block found in CSV; skipping summary_stats.png")
        return

    stat_order = [s for s in ["min", "median", "mean", "max", "stddev"] if s in stats]
    gpu_vals = ns_to_ms([stats[s]["gpu_ns"] for s in stat_order])
    cpu_vals = ns_to_ms([stats[s]["cpu_ns"] for s in stat_order])

    x = range(len(stat_order))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], gpu_vals, width, label="GPU")
    plt.bar([i + width / 2 for i in x], cpu_vals, width, label="CPU")
    plt.xticks(list(x), stat_order)
    plt.ylabel("Time (ms)")
    plt.title("Summary Statistics: GPU vs CPU")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "summary_stats.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_boxplot(gpu_times, cpu_times, out_dir):
    gpu_ms = ns_to_ms(gpu_times)
    cpu_ms = ns_to_ms(cpu_times)

    plt.figure(figsize=(6, 5))
    plt.boxplot([gpu_ms, cpu_ms], tick_labels=["GPU", "CPU"])
    plt.ylabel("Time (ms)")
    plt.title("Distribution of Execution Times")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "boxplot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) >= 2 else "statistics.csv"
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "."

    if not os.path.isfile(csv_path):
        print(f"Error: could not find '{csv_path}'")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    iterations, gpu_times, cpu_times, stats = parse_statistics_csv(csv_path)

    if not iterations:
        print("Error: no per-iteration rows found in the CSV.")
        sys.exit(1)

    plot_timeline(iterations, gpu_times, cpu_times, out_dir)
    plot_speedup(iterations, gpu_times, cpu_times, out_dir)
    plot_summary_stats(stats, out_dir)
    plot_boxplot(gpu_times, cpu_times, out_dir)

    print("\nGenerated 4 graphs in:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
