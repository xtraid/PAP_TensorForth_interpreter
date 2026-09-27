# TensorForth vs OpenBLAS: prepared-input matmul

This harness compares the existing TensorForth `@` operation with OpenBLAS
`cblas_sgemm` on identical, contiguous row-major float32 matrices. It does not
modify the interpreter or its multiplication algorithm.

## Build and run

Requires GCC, OpenMP, and OpenBLAS development headers/library (not reference
BLAS). From the repository root:

```sh
make -C benchmarks
bash benchmarks/run.sh 5000 5 12
```

For the local unpacked OpenBLAS installation used for the recorded run:

```sh
make -C benchmarks OPENBLAS_PREFIX=.deps/openblas/usr
bash benchmarks/run.sh 5000 5 12
```

`OPENBLAS_PREFIX` is relative to `benchmarks/`. Runtime objects are built
separately from the normal interpreter, using its `PERF=1` optimization flags:
`-O3 -march=native -ffast-math -funroll-loops`, C17 and OpenMP. The harness and
validation code use `-O3` without fast-math. If changing compiler flags or the
library installation, use `make -B -C benchmarks ...` to force a fresh build.

Arguments are matrix order, number of measured repetitions, and thread count.
Each run writes a unique log under `results/`, preserving raw samples, medians,
compiler and library identity, source/binary hashes, and thread configuration.
The shell script fails if validation or execution fails; incomplete logs must
not be treated as successful benchmarks.

## Exactly what is timed

Inputs A and B are allocated, generated and touched once **before any timer**.
The deterministic xorshift32 generator starts at seed 123456789 and fills A,
then B, with values in [0, 1). Both implementations read those same buffers.
The OpenBLAS C output buffer is allocated and touched before timing.

| TensorForth | OpenBLAS |
| --- | --- |
| Time the real `parser("@", stack)` call | Time the real `cblas_sgemm(...)` call |
| Prepare stack before timer; B below A | A, B and C already available |
| Includes dispatch, output allocation, operand transpose, temporary allocation/free and parallel multiply inside `@` | Includes all internal work of SGEMM; C is supplied by the caller |
| Stop before popping/freeing output (`D` is not timed) | Stop when SGEMM returns |

This is an **operation/API comparison with prepared inputs**, not an isolated
arithmetic-kernel comparison or an end-to-end script benchmark. Input random
generation, stack setup, output validation, printing and final cleanup are
excluded. TensorForth's internal allocation and transpose remain included
because they are part of the production operator. Do not describe both sides
as using preallocated output buffers or as timing identical internal work.

The operand order matters: TensorForth computes TOS × second-from-top, so the
harness pushes B and then A to match OpenBLAS A × B. `A B @` is not valid named
variable syntax in this repository. The thumbnail can show the real preparation
code `[ 5000 5000 ] ?` twice, but must distinguish that from the timed `@`.

One complete warm-up per implementation is excluded. Five measured rounds
alternate call order. No cache flushing; this is repeated use of prepared
inputs. Report median elapsed wall time measured with `CLOCK_MONOTONIC` and
`GFLOP/s = 2 × N³ / seconds / 10⁹`. At N=5000 the numerator is 250 GFLOP.
No speed-ratio claim is required.

The thread count is explicitly set and checked for both libraries. The runner
sets `OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=FALSE`, `OMP_WAIT_POLICY=PASSIVE`, and
both thread-count environment variables. There is no CPU pinning or power-mode
change. These are local laptop measurements, affected by thermal state and
other running applications; they are not portable performance guarantees.

## Correctness checks

Every round, including warm-up, compares every output element, requiring finite
values and `abs(TF - BLAS) <= 1e-4 + 1e-4 * abs(BLAS)`. Sixteen deterministic
positions are independently recomputed with double-precision accumulation and
checked against both outputs using the same absolute/relative tolerances.
All validation is outside timing. Small runs at N=7 and N=65 exercise the small
matrix path and blocked path with a partial boundary tile respectively.

OpenBLAS API reference: <https://www.openmathlib.org/OpenBLAS/docs/faq/>.
