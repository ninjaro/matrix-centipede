#!/bin/bash -l
#SBATCH --job-name=bench
#SBATCH --partition=c23ms_low
#SBATCH --time=06:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

set -euo pipefail

module purge || true
module load GCCcore/13.3.0
module load CMake/3.29.3
module load Python/3.11

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MAX_N="${MAX_N:-1024}"
REPEATS="${REPEATS:-1}"

VENV="$HOME/.venvs/py311-mpl"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install --no-cache-dir matplotlib
fi
export MPLBACKEND=Agg

rm -rf build
mkdir -p bench
rm -f bench/bench_log.txt bench/bench_plot.png

cmake -S . -B build \
  -DPython3_EXECUTABLE="$VENV/bin/python" \
  -DBUILD_BENCHMARKS=ON \
  -DDM_BENCH_MAX_N="${MAX_N}" \
  -DDM_BENCH_REPETITIONS="${REPEATS}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target bench_plot -- -j ${SLURM_CPUS_PER_TASK:-1}

