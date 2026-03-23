#!/usr/bin/env python3
"""
plot_3d.py — 3-D surface plots for the bench_3d n_queries × n_db sweep.

Reads the JSON produced by bench_3d and generates three PNG files:

  3d_nontiled_cpu_vs_gpu.png  — CPU non-tiled  vs GPU non-tiled
  3d_tiled_cpu_vs_gpu.png     — CPU tiled       vs GPU tiled
  3d_gpu_nontiled_vs_tiled.png— GPU non-tiled   vs GPU tiled

Each plot has X = log₂(n_queries), Y = log₁₀(n_db), Z = log₁₀(batch_time_ms).
Missing / skipped cells (null JSON values) appear as gaps in the surface.

Usage:
    python3 plot_3d.py [bench_3d_results.json] [--out-dir plots_3d/]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str):
    with open(path) as fh:
        data = json.load(fh)
    return data


def build_grid(records: list[dict], backend: str,
               nq_list: list[int], nd_list: list[int]) -> np.ndarray:
    """
    Return a (len(nd_list) × len(nq_list)) array of batch_time_ms values.
    Missing / null entries are NaN.
    """
    lookup = {}
    for r in records:
        if r["backend"] == backend and r["batch_time_ms"] is not None:
            lookup[(r["n_queries"], r["n_db"])] = r["batch_time_ms"]

    Z = np.full((len(nd_list), len(nq_list)), np.nan)
    for j, nq in enumerate(nq_list):
        for i, nd in enumerate(nd_list):
            v = lookup.get((nq, nd))
            if v is not None and v > 0:
                Z[i, j] = v
    return Z


def human_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── core plot function ────────────────────────────────────────────────────────

SURFACE_ALPHA = 0.75

STYLE = {
    "cpu_nontiled": dict(color="#2196F3", label="CPU non-tiled (DenseIndex)"),
    "cpu_tiled":    dict(color="#FF9800", label="CPU tiled (TiledDenseIndex)"),
    "gpu_nontiled": dict(color="#E91E63", label="GPU non-tiled (warp-per-row)"),
    "gpu_tiled":    dict(color="#4CAF50", label="GPU tiled (shared-mem tiles)"),
}


def make_surface(ax, X, Y, Z_ms, color, label, alpha=SURFACE_ALPHA):
    """
    Plot log10(Z_ms) as a coloured surface.  NaN cells are masked (gap).
    """
    Z_log = np.where(np.isfinite(Z_ms) & (Z_ms > 0), np.log10(Z_ms), np.nan)
    masked = np.ma.masked_invalid(Z_log)
    ax.plot_surface(X, Y, masked,
                    color=color, alpha=alpha, linewidth=0.3, edgecolor="white",
                    label=label)
    return Z_log


def format_axes(ax, nq_list, nd_list, title: str):
    ax.set_xlabel("n_queries", labelpad=8)
    ax.set_ylabel("n_db",      labelpad=8)
    ax.set_zlabel("batch time (ms, log₁₀ scale)", labelpad=8)
    ax.set_title(title, pad=12)

    # X ticks: log2(n_queries)
    ax.set_xticks(np.log2(nq_list))
    ax.set_xticklabels([str(n) for n in nq_list], fontsize=8)

    # Y ticks: log10(n_db)
    ax.set_yticks(np.log10(nd_list))
    ax.set_yticklabels([human_n(n) for n in nd_list], fontsize=8)

    # Z ticks: fixed powers of 10 from 0.001 ms to 10 000 ms
    z_powers = np.arange(-3, 5)        # 10^-3 .. 10^4
    ax.set_zticks(z_powers)
    ax.set_zticklabels([f"$10^{{{p}}}$" for p in z_powers], fontsize=7)

    ax.view_init(elev=25, azim=-50)


def comparison_plot(records, nq_list, nd_list, out_path,
                    backend_a, backend_b, title):
    """
    Two-surface 3D plot comparing backend_a against backend_b.
    """
    X, Y = np.meshgrid(np.log2(nq_list), np.log10(nd_list))

    Za = build_grid(records, backend_a, nq_list, nd_list)
    Zb = build_grid(records, backend_b, nq_list, nd_list)

    has_a = not np.all(np.isnan(Za))
    has_b = not np.all(np.isnan(Zb))

    if not has_a and not has_b:
        print(f"  [skip] no data for {backend_a} or {backend_b}")
        return

    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection="3d")

    if has_a:
        make_surface(ax, X, Y, Za, STYLE[backend_a]["color"], STYLE[backend_a]["label"])
    if has_b:
        make_surface(ax, X, Y, Zb, STYLE[backend_b]["color"], STYLE[backend_b]["label"])

    # Add a "speedup" annotation: where is backend_b faster than backend_a?
    if has_a and has_b:
        speedup = Za / Zb   # > 1 means b is faster, < 1 means a is faster
        n_faster = int(np.nansum(speedup > 1))
        n_total  = int(np.sum(np.isfinite(speedup)))
        ax.text2D(
            0.02, 0.97,
            f"{STYLE[backend_b]['label'].split('(')[0].strip()} faster in "
            f"{n_faster}/{n_total} cells",
            transform=ax.transAxes, fontsize=8, va="top",
            color=STYLE[backend_b]["color"],
        )

    format_axes(ax, nq_list, nd_list, title)

    # Legend via proxy patches (plot_surface labels not shown by default)
    from matplotlib.patches import Patch
    handles = []
    if has_a:
        handles.append(Patch(color=STYLE[backend_a]["color"],
                             label=STYLE[backend_a]["label"], alpha=SURFACE_ALPHA))
    if has_b:
        handles.append(Patch(color=STYLE[backend_b]["color"],
                             label=STYLE[backend_b]["label"], alpha=SURFACE_ALPHA))
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    save(fig, out_path)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", default="bench_3d_results.json",
                    help="Path to bench_3d JSON (default: bench_3d_results.json)")
    ap.add_argument("--out-dir", default="plots_3d",
                    help="Output directory for PNG files (default: plots_3d/)")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        print(f"ERROR: {args.json} not found.  Run bench_3d first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    data     = load(args.json)
    records  = data["results"]
    nq_list  = data.get("n_queries_list", [1, 4, 16, 64, 256, 512])
    nd_list  = data.get("n_db_list",      [10000, 50000, 100000, 500000])
    gpu_name = data.get("gpu_name", "")

    gpu_suffix = f" [{gpu_name}]" if gpu_name else ""

    print(f"Loaded {len(records)} records from {args.json}")
    print(f"Generating 3D plots in {args.out_dir}/")

    # ── Plot 1: CPU non-tiled vs GPU non-tiled ──────────────────────────────
    comparison_plot(
        records, nq_list, nd_list,
        os.path.join(args.out_dir, "3d_nontiled_cpu_vs_gpu.png"),
        backend_a="cpu_nontiled",
        backend_b="gpu_nontiled",
        title=f"Non-tiled CPU vs Non-tiled GPU (L2, dim=128){gpu_suffix}\n"
              "Lower surface = faster",
    )

    # ── Plot 2: CPU tiled vs GPU tiled ──────────────────────────────────────
    comparison_plot(
        records, nq_list, nd_list,
        os.path.join(args.out_dir, "3d_tiled_cpu_vs_gpu.png"),
        backend_a="cpu_tiled",
        backend_b="gpu_tiled",
        title=f"Tiled CPU vs Tiled GPU (L2, dim=128){gpu_suffix}\n"
              "Lower surface = faster",
    )

    # ── Plot 3: GPU non-tiled vs GPU tiled ──────────────────────────────────
    comparison_plot(
        records, nq_list, nd_list,
        os.path.join(args.out_dir, "3d_gpu_nontiled_vs_tiled.png"),
        backend_a="gpu_nontiled",
        backend_b="gpu_tiled",
        title=f"GPU non-tiled vs GPU tiled (L2, dim=128){gpu_suffix}\n"
              "Effect of shared-memory tiling",
    )

    print(f"\nDone — 3 plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
