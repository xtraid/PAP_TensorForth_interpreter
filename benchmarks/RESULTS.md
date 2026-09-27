# Recorded 5000×5000 matmul comparison

Measured 2026-09-27 on Intel Core i7-1255U (12 logical CPUs), Linux
7.2.5-3-omarchy, GCC 16.2.1. TensorForth source commit
`6ed061f0e3eae19aab4213914a6314406cdb73a8`, performance compiler flags documented
in [README.md](README.md). OpenBLAS 0.3.34, dynamic Haswell kernel, OpenMP build.
Both used 12 threads. One excluded warm-up, five measured repetitions.

| Operation | Median time | GFLOP/s | Observed time range |
| --- | ---: | ---: | ---: |
| TensorForth `@` | 6.850892921 s | 36.491594 | 6.171163908–7.319124465 s |
| OpenBLAS `cblas_sgemm` | 2.286164988 s | 109.353437 | 2.088275829–3.346330381 s |

**Prepared inputs; public-operation timing, not isolated kernels.** TensorForth
includes dispatch, result allocation and operand transpose inside `@`.
OpenBLAS receives preallocated C. Generation, stack preparation, validation,
printing and output destruction are outside timing. The same A and B buffers
are used on both sides. See [methodology](README.md) before reusing these numbers.

All six rounds passed the full output comparison and 16 independently
accumulated double-precision reference elements. Maximum relative difference
between TensorForth and OpenBLAS: `8.80978097e-7`. Validation tolerance:
`1e-4 + 1e-4 * abs(reference)`.

[Raw log with all samples and source hashes](results/2026-09-27-i7-1255U-5000x5000.log).

Thumbnail values rounded from the median: **6.851 s / 36.49 GFLOP/s** and
**2.286 s / 109.35 GFLOP/s**. These are local laptop observations; the recorded
spread is substantial, especially for OpenBLAS, and no speedup claim is made.

The OpenBLAS package was extracted locally (no system package changes):
`openblas-0.3.34-1-x86_64.pkg.tar.zst`, SHA-256
`854099273915eadb8d5e94e88104c4a14fe26a275ecd71d4590cdbc8545765a3`.
