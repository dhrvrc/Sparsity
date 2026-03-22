#!/usr/bin/env python3
"""
plot_results.py — visualise bench_dense output.

Usage:
    python plot_results.py [bench_results.json] [--out-dir plots/]

Generates seven figures (PNG, 150 dpi) in the output directory:
  1. latency_vs_ndb.png        — p50 latency vs DB size  (dim=128)
  2. latency_vs_dim.png        — p50 latency vs dimension (n_db=100K)
  3. qps_comparison.png        — QPS bar chart            (dim=128, n_db=100K)
  4. recall_vs_qps.png         — IVF recall/speed curve   (n_probe sweep)
  5. build_time.png            — build time vs DB size    (dim=128)
  6. memory.png                — memory vs DB size        (dim=128)
  7. batch_tiling.png          — QPS vs batch size        (tiling benefit)
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── colour / style constants ──────────────────────────────────────────────────

COLOURS = {
    "DenseIndex":      "#2196F3",   # blue
    "TiledDenseIndex": "#FF9800",   # orange
    "IVFIndex":        "#4CAF50",   # green
}
MARKERS = {
    "DenseIndex":      "o",
    "TiledDenseIndex": "s",
    "IVFIndex":        "^",
}
LINE_STYLES = {
    "gaussian":  "-",
    "clustered": "--",
}

plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> list[dict]:
    with open(path) as fh:
        data = json.load(fh)
    return data["results"]


def filter_rows(rows, **kwargs):
    """Return rows matching all keyword conditions (exact equality)."""
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in kwargs.items()):
            out.append(r)
    return out


def human_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── plot 1: latency vs DB size ────────────────────────────────────────────────

def plot_latency_vs_ndb(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, metric in zip(axes, ["L2", "Cosine"]):
        subset = filter_rows(rows, dim=128, metric=metric, distribution="gaussian",
                             batch_size=None)
        # Only scaling records have batch_size == null (-1 in C++, serialised as null → None)
        subset = [r for r in subset if r.get("batch_size") is None
                  and r["index"] in ("DenseIndex", "TiledDenseIndex")]

        for idx_name in ("DenseIndex", "TiledDenseIndex"):
            pts = sorted([r for r in subset if r["index"] == idx_name],
                         key=lambda r: r["n_db"])
            if not pts:
                continue
            xs = [p["n_db"] for p in pts]
            ys = [p["p50_us"] for p in pts]
            ax.plot(xs, ys,
                    color=COLOURS[idx_name],
                    marker=MARKERS[idx_name],
                    label=idx_name,
                    linewidth=2, markersize=7)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("DB size (vectors)")
        ax.set_ylabel("Latency p50 (µs)")
        ax.set_title(f"Single-query latency vs DB size [{metric}] — dim=128")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_n(int(x))))
        ax.legend()

    save(fig, os.path.join(out_dir, "latency_vs_ndb.png"))


# ── plot 2: latency vs dimension ─────────────────────────────────────────────

def plot_latency_vs_dim(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, metric in zip(axes, ["L2", "Cosine"]):
        subset = [r for r in rows
                  if r["n_db"] == 100000
                  and r["metric"] == metric
                  and r["distribution"] == "gaussian"
                  and r.get("batch_size") is None
                  and r["index"] in ("DenseIndex", "TiledDenseIndex")]

        for idx_name in ("DenseIndex", "TiledDenseIndex"):
            pts = sorted([r for r in subset if r["index"] == idx_name],
                         key=lambda r: r["dim"])
            if not pts:
                continue
            xs = [p["dim"] for p in pts]
            ys = [p["p50_us"] for p in pts]
            ax.plot(xs, ys,
                    color=COLOURS[idx_name],
                    marker=MARKERS[idx_name],
                    label=idx_name,
                    linewidth=2, markersize=7)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Vector dimension")
        ax.set_ylabel("Latency p50 (µs)")
        ax.set_title(f"Single-query latency vs dimension [{metric}] — n_db=100K")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend()

    save(fig, os.path.join(out_dir, "latency_vs_dim.png"))


# ── plot 3: QPS comparison bar chart ─────────────────────────────────────────

def plot_qps_comparison(rows, out_dir):
    indices  = ["DenseIndex", "TiledDenseIndex"]
    metrics  = ["L2", "Cosine"]
    n_db     = 100000
    dim      = 128

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, metric in zip(axes, metrics):
        vals = {}
        for idx_name in indices:
            pts = [r for r in rows
                   if r["index"] == idx_name
                   and r["metric"] == metric
                   and r["n_db"] == n_db
                   and r["dim"] == dim
                   and r["distribution"] == "gaussian"
                   and r.get("batch_size") is None]
            vals[idx_name] = pts[0]["qps"] if pts else 0.0

        colours = [COLOURS[n] for n in indices]
        bars = ax.bar(indices, [vals[n] for n in indices], color=colours, width=0.5, edgecolor="black")

        for bar, idx_name in zip(bars, indices):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f"{h:,.0f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Queries per second (QPS)")
        ax.set_title(f"Throughput comparison [{metric}]\n(dim={dim}, n_db={human_n(n_db)}, batch={BATCH_QPS_SIZE})")
        ax.set_ylim(0, max(vals.values()) * 1.25 if vals else 1)

    save(fig, os.path.join(out_dir, "qps_comparison.png"))


# ── plot 4: IVF recall vs QPS ────────────────────────────────────────────────

def plot_recall_vs_qps(rows, out_dir):
    ivf_rows = [r for r in rows if r["index"] == "IVFIndex"]
    if not ivf_rows:
        print("  [skip] no IVF records found")
        return

    metrics  = sorted(set(r["metric"]       for r in ivf_rows))
    n_dbs    = sorted(set(r["n_db"]         for r in ivf_rows))
    dists    = sorted(set(r["distribution"] for r in ivf_rows))

    n_rows = len(metrics)
    n_cols = len(n_dbs)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    dist_colours = {"gaussian": "#2196F3", "clustered": "#E91E63"}
    dist_labels  = {"gaussian": "gaussian (random)", "clustered": "clustered (10 blobs)"}

    for row_i, metric in enumerate(metrics):
        for col_i, n_db in enumerate(n_dbs):
            ax = axes[row_i][col_i]
            for dist in dists:
                pts = sorted(
                    [r for r in ivf_rows
                     if r["metric"] == metric
                     and r["n_db"] == n_db
                     and r["distribution"] == dist],
                    key=lambda r: r["n_probe"])
                if not pts:
                    continue
                xs = [p["qps"]    for p in pts]
                ys = [p["recall"] for p in pts]
                ax.plot(xs, ys,
                        color=dist_colours[dist],
                        linestyle=LINE_STYLES.get(dist, "-"),
                        marker="o", markersize=6,
                        label=dist_labels[dist],
                        linewidth=2)
                # annotate key n_probe values
                for p in pts:
                    if p["n_probe"] in (1, p["n_lists"]):
                        ax.annotate(f"np={p['n_probe']}",
                                    (p["qps"], p["recall"]),
                                    textcoords="offset points",
                                    xytext=(6, 4), fontsize=8)

            ax.set_xscale("log")
            ax.set_xlabel("Queries per second (QPS)")
            ax.set_ylabel("Recall@10")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"IVF recall vs speed [{metric}]\n(dim=128, n_db={human_n(n_db)})")
            ax.legend()

    save(fig, os.path.join(out_dir, "recall_vs_qps.png"))


# ── plot 5: build time vs DB size ────────────────────────────────────────────

def plot_build_time(rows, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    indices = ["DenseIndex", "TiledDenseIndex", "IVFIndex"]
    seen    = set()

    for idx_name in indices:
        # For IVFIndex: one representative build time per n_db (pick n_probe=1, gaussian)
        if idx_name == "IVFIndex":
            pts = sorted(
                [r for r in rows
                 if r["index"] == "IVFIndex"
                 and r["metric"] == "L2"
                 and r["distribution"] == "gaussian"
                 and r["dim"] == 128
                 and r["n_probe"] == 1],   # build_ms is shared; take first n_probe entry
                key=lambda r: r["n_db"])
        else:
            pts = sorted(
                [r for r in rows
                 if r["index"] == idx_name
                 and r["metric"] == "L2"
                 and r["distribution"] == "gaussian"
                 and r["dim"] == 128
                 and r.get("batch_size") is None],
                key=lambda r: r["n_db"])

        if not pts:
            continue
        xs = [p["n_db"]    for p in pts]
        ys = [p["build_ms"] for p in pts]
        ax.plot(xs, ys,
                color=COLOURS.get(idx_name, "grey"),
                marker=MARKERS.get(idx_name, "x"),
                label=idx_name,
                linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DB size (vectors)")
    ax.set_ylabel("Build time (ms)")
    ax.set_title("Index build time vs DB size [L2, dim=128]")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_n(int(x))))
    ax.legend()

    save(fig, os.path.join(out_dir, "build_time.png"))


# ── plot 6: memory vs DB size ────────────────────────────────────────────────

def plot_memory(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    indices = ["DenseIndex", "TiledDenseIndex", "IVFIndex"]

    # Left: measured RSS delta
    ax = axes[0]
    for idx_name in indices:
        if idx_name == "IVFIndex":
            pts = sorted(
                [r for r in rows
                 if r["index"] == "IVFIndex"
                 and r["metric"] == "L2"
                 and r["distribution"] == "gaussian"
                 and r["dim"] == 128
                 and r["n_probe"] == 1],
                key=lambda r: r["n_db"])
        else:
            pts = sorted(
                [r for r in rows
                 if r["index"] == idx_name
                 and r["metric"] == "L2"
                 and r["distribution"] == "gaussian"
                 and r["dim"] == 128
                 and r.get("batch_size") is None],
                key=lambda r: r["n_db"])
        if not pts:
            continue
        xs = [p["n_db"]                  for p in pts]
        ys = [p["build_rss_delta_kb"] / 1024 for p in pts]  # → MB
        ax.plot(xs, ys, color=COLOURS.get(idx_name, "grey"),
                marker=MARKERS.get(idx_name, "x"),
                label=idx_name, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("DB size (vectors)")
    ax.set_ylabel("RSS delta (MB)")
    ax.set_title("Measured memory footprint [L2, dim=128]\n(RSS delta; approx — includes OS overheads)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_n(int(x))))
    ax.legend()

    # Right: theoretical bytes
    ax = axes[1]
    for idx_name in indices:
        if idx_name == "IVFIndex":
            pts = sorted(
                [r for r in rows
                 if r["index"] == "IVFIndex"
                 and r["metric"] == "L2"
                 and r["distribution"] == "gaussian"
                 and r["dim"] == 128
                 and r["n_probe"] == 1],
                key=lambda r: r["n_db"])
        else:
            pts = sorted(
                [r for r in rows
                 if r["index"] == idx_name
                 and r["metric"] == "L2"
                 and r["distribution"] == "gaussian"
                 and r["dim"] == 128
                 and r.get("batch_size") is None],
                key=lambda r: r["n_db"])
        if not pts:
            continue
        xs = [p["n_db"]                          for p in pts]
        ys = [p["theoretical_bytes"] / 1024 / 1024 for p in pts]  # → MB
        ax.plot(xs, ys, color=COLOURS.get(idx_name, "grey"),
                marker=MARKERS.get(idx_name, "x"),
                label=idx_name, linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("DB size (vectors)")
    ax.set_ylabel("Theoretical storage (MB)")
    ax.set_title("Theoretical index storage [L2, dim=128]\n(vectors + norms + centroids + IDs)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_n(int(x))))
    ax.legend()

    save(fig, os.path.join(out_dir, "memory.png"))


# ── plot 7: batch size / tiling benefit ──────────────────────────────────────

def plot_batch_tiling(rows, out_dir):
    batch_rows = [r for r in rows if r.get("batch_size") is not None]
    if not batch_rows:
        print("  [skip] no batch-sweep records found")
        return

    dims_present = sorted(set(r["dim"] for r in batch_rows))
    fig, axes = plt.subplots(1, len(dims_present),
                              figsize=(7 * len(dims_present), 4.5),
                              squeeze=False)

    for ax, dim in zip(axes[0], dims_present):
        for idx_name in ("DenseIndex", "TiledDenseIndex"):
            pts = sorted(
                [r for r in batch_rows
                 if r["index"] == idx_name and r["dim"] == dim],
                key=lambda r: r["batch_size"])
            if not pts:
                continue
            xs = [p["batch_size"] for p in pts]
            ys = [p["qps"]        for p in pts]
            ax.plot(xs, ys,
                    color=COLOURS[idx_name],
                    marker=MARKERS[idx_name],
                    label=idx_name,
                    linewidth=2, markersize=7)

        # Mark TILE_Q=4 boundary
        ax.axvline(x=4, color="grey", linestyle=":", alpha=0.7, label="TILE_Q=4")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Query batch size")
        ax.set_ylabel("Queries per second (QPS)")
        ax.set_title(f"Tiling benefit — dim={dim}, n_db=100K [L2]")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend()

    save(fig, os.path.join(out_dir, "batch_tiling.png"))


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(rows):
    print("\n── Summary (dim=128, n_db=100K, L2, gaussian) ────────────────────")
    print(f"{'Index':<20} {'Build ms':>10} {'p50 µs':>10} {'QPS':>10} {'Recall':>8}")
    print("-" * 62)
    for idx_name in ("DenseIndex", "TiledDenseIndex", "IVFIndex"):
        if idx_name == "IVFIndex":
            pts = [r for r in rows
                   if r["index"] == "IVFIndex"
                   and r["metric"] == "L2"
                   and r["distribution"] == "gaussian"
                   and r["dim"] == 128
                   and r["n_db"] == 100000]
            if not pts:
                continue
            # Show min, mid, max n_probe rows
            pts = sorted(pts, key=lambda r: r["n_probe"])
            for p in [pts[0], pts[len(pts) // 2], pts[-1]]:
                label = f"IVF np={p['n_probe']}"
                print(f"  {label:<18} {p['build_ms']:>10.1f} {p['p50_us']:>10.1f} "
                      f"{p['qps']:>10.0f} {p['recall']:>8.3f}")
        else:
            pts = [r for r in rows
                   if r["index"] == idx_name
                   and r["metric"] == "L2"
                   and r["distribution"] == "gaussian"
                   and r["dim"] == 128
                   and r["n_db"] == 100000
                   and r.get("batch_size") is None]
            if not pts:
                continue
            p = pts[0]
            print(f"  {idx_name:<18} {p['build_ms']:>10.1f} {p['p50_us']:>10.1f} "
                  f"{p['qps']:>10.0f} {p['recall']:>8.3f}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────

# Keep BATCH_QPS_SIZE in sync with bench_dense.cpp for the QPS bar-chart title
BATCH_QPS_SIZE = 500

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", default="bench_results.json",
                    help="Path to the benchmark JSON file (default: bench_results.json)")
    ap.add_argument("--out-dir", default="plots",
                    help="Directory to write PNG files into (default: plots/)")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"ERROR: {args.json} not found. Run bench_dense first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(args.json)
    print(f"Loaded {len(rows)} records from {args.json}")

    print_summary(rows)

    print("Generating plots...")
    plot_latency_vs_ndb(rows, args.out_dir)
    plot_latency_vs_dim(rows, args.out_dir)
    plot_qps_comparison(rows, args.out_dir)
    plot_recall_vs_qps(rows, args.out_dir)
    plot_build_time(rows, args.out_dir)
    plot_memory(rows, args.out_dir)
    plot_batch_tiling(rows, args.out_dir)

    print(f"\nDone — plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
