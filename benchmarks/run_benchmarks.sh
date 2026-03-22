#!/usr/bin/env bash
# run_benchmarks.sh — build and run the dense benchmarks, then plot results.
#
# Usage (from the repo root or benchmarks/):
#   ./benchmarks/run_benchmarks.sh [--out results/my_run.json] [--build-dir build]
#
# Options:
#   --build-dir DIR   CMake build directory (default: build)
#   --out FILE        JSON output path      (default: benchmarks/results/bench_TIMESTAMP.json)
#   --plot-only FILE  Skip build+bench; just re-plot an existing JSON file
#   --quick           Reduce to a subset of dims/n_dbs for a faster test run
#   --no-cuda         Disable CUDA even if nvcc is found

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_ROOT/benchmarks/results"
OUT_JSON=""
PLOTS_DIR=""
PLOT_ONLY=""
FORCE_NO_CUDA=0

# ── parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --out)       OUT_JSON="$2";  shift 2 ;;
        --plot-only) PLOT_ONLY="$2"; shift 2 ;;
        --quick)     QUICK=1; shift ;;
        --no-cuda)   FORCE_NO_CUDA=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT_JSON" ]]; then
    mkdir -p "$RESULTS_DIR"
    OUT_JSON="$RESULTS_DIR/bench_${TIMESTAMP}.json"
fi
PLOTS_DIR="${OUT_JSON%.json}_plots"

# ── plot-only mode ────────────────────────────────────────────────────────────
if [[ -n "$PLOT_ONLY" ]]; then
    echo "── Plot only: $PLOT_ONLY ──"
    python3 "$REPO_ROOT/benchmarks/plot_results.py" "$PLOT_ONLY" --out-dir "$PLOTS_DIR"
    echo "Plots written to: $PLOTS_DIR/"
    exit 0
fi

# ── detect CUDA ───────────────────────────────────────────────────────────────
if [[ "$FORCE_NO_CUDA" -eq 1 ]]; then
    WITH_CUDA=OFF
    echo "CUDA: disabled by --no-cuda"
elif command -v nvcc &>/dev/null; then
    WITH_CUDA=ON
    echo "CUDA: nvcc found ($(nvcc --version | grep release | awk '{print $5}' | tr -d ,)) — GPU benchmarks enabled"
else
    WITH_CUDA=OFF
    echo "CUDA: nvcc not found — bench_gpu will record CPU numbers only"
    echo "      Install the CUDA toolkit and re-run to enable GPU benchmarks"
fi

# ── configure and build ───────────────────────────────────────────────────────
echo
echo "── Build ────────────────────────────────────────────────────────────"
echo "  repo root  : $REPO_ROOT"
echo "  build dir  : $BUILD_DIR"
echo "  WITH_CUDA  : $WITH_CUDA"
echo "  output     : $OUT_JSON"
echo "  plots      : $PLOTS_DIR/"
echo

# Reconfigure if: no cache yet, or the WITH_CUDA setting differs from the cache.
CACHED_CUDA=""
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    CACHED_CUDA=$(grep -m1 "^WITH_CUDA:" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null \
                  | cut -d= -f2 | tr -d '[:space:]' || true)
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "[cmake configure — WITH_CUDA=$WITH_CUDA]"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_BENCHMARKS=ON -DBUILD_TESTS=OFF -DWITH_CUDA="$WITH_CUDA"
elif [[ "$CACHED_CUDA" != "$WITH_CUDA" ]]; then
    echo "[cmake reconfigure — WITH_CUDA changed: $CACHED_CUDA → $WITH_CUDA]"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_BENCHMARKS=ON -DBUILD_TESTS=OFF -DWITH_CUDA="$WITH_CUDA"
else
    echo "[cmake configure — using existing cache (WITH_CUDA=$CACHED_CUDA)]"
fi

echo "[cmake build — bench_dense + bench_gpu]"
cmake --build "$BUILD_DIR" --target bench_dense bench_gpu --parallel "$(nproc)"

BENCH_BIN="$BUILD_DIR/benchmarks/bench_dense"
GPU_BIN="$BUILD_DIR/benchmarks/bench_gpu"

if [[ ! -x "$BENCH_BIN" ]]; then
    echo "ERROR: bench_dense binary not found at $BENCH_BIN" >&2
    exit 1
fi
if [[ ! -x "$GPU_BIN" ]]; then
    echo "ERROR: bench_gpu binary not found at $GPU_BIN" >&2
    exit 1
fi

GPU_JSON="${OUT_JSON%.json}_gpu.json"

# ── run benchmarks ────────────────────────────────────────────────────────────
echo
echo "── Run: bench_dense ─────────────────────────────────────────────────"
echo "  binary : $BENCH_BIN"
echo "  output : $OUT_JSON"
echo

SECONDS=0
"$BENCH_BIN" "$OUT_JSON"
ELAPSED=$SECONDS
echo
echo "bench_dense finished in ${ELAPSED}s"

echo
echo "── Run: bench_gpu ───────────────────────────────────────────────────"
echo "  binary : $GPU_BIN"
echo "  output : $GPU_JSON"
echo

SECONDS=0
"$GPU_BIN" "$GPU_JSON"
ELAPSED=$SECONDS
echo
echo "bench_gpu finished in ${ELAPSED}s"

# ── plot ──────────────────────────────────────────────────────────────────────
echo
echo "── Plot ─────────────────────────────────────────────────────────────"
mkdir -p "$PLOTS_DIR"

if command -v python3 &>/dev/null && python3 -c "import matplotlib" 2>/dev/null; then
    python3 "$REPO_ROOT/benchmarks/plot_results.py" "$OUT_JSON" --out-dir "$PLOTS_DIR"
    echo "Plots written to: $PLOTS_DIR/"
else
    echo "matplotlib not found — skipping plots."
    echo "Install with:  pip install matplotlib"
    echo "Then run:      python3 benchmarks/plot_results.py $OUT_JSON --out-dir $PLOTS_DIR"
fi

echo
echo "Done."
echo "  Dense JSON : $OUT_JSON"
echo "  GPU JSON   : $GPU_JSON"
echo "  Plots      : $PLOTS_DIR/"
