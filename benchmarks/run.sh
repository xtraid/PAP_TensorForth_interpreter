#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p results
result=$(mktemp "results/matmul-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
# Identical explicit OpenMP policy for this OpenMP build of OpenBLAS and TF.
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS="${3:-12}" OPENBLAS_NUM_THREADS="${3:-12}"
{
    date -u --iso-8601=seconds
    uname -srmo
    lscpu | sed -n '/^Model name:/p; /^CPU(s):/p'
    gcc --version | head -1
    git rev-parse HEAD
    sha256sum matmul_bench.c Makefile run.sh ../*.c ../*.h build/matmul_bench
    ldd build/matmul_bench
    printf 'OMP_DYNAMIC=%s OMP_PROC_BIND=%s OMP_WAIT_POLICY=%s\n' \
        "$OMP_DYNAMIC" "$OMP_PROC_BIND" "$OMP_WAIT_POLICY"
    printf 'command: benchmarks/build/matmul_bench %s %s %s\n' \
        "${1:-5000}" "${2:-5}" "${3:-12}"
    build/matmul_bench "${1:-5000}" "${2:-5}" "${3:-12}"
} 2>&1 | tee "$result"
printf 'Saved: %s/%s\n' "$PWD" "$result"
