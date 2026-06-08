# TensorForth Interpreter

A stack-based interpreter for tensor computation, written in C17. TensorForth is a Forth-inspired language specialized for linear algebra operations on 2D tensors (matrices).

## Overview

TensorForth programs are sequences of single-character opcodes that operate on a stack of tensors. All values on the stack are 2D arrays (1D arrays are represented as row vectors with `shape = [1, N]`). Operations consume their operands from the stack and push results back.

## Building

Requires `gcc` with C17 support and `OpenMP`.

```sh
make              # optimized build (-O3)
make DEBUG=1      # debug build (-O0 -g)
make PERF=1       # performance build (-O3 -march=native -ffast-math -funroll-loops)
make clean        # remove build artifacts
```

This produces a `tensorforth` executable.

## Usage

```sh
./tensorforth <script.tensorforth>
./run_tests.sh            # run the test suite (82 tests)
./run_tests.sh --valgrind # run with valgrind leak check
```

## Language Reference

### Tensor Literal

```
[ v1 v2 ... vN ]
```

Pushes a `[1 × N]` tensor onto the stack with the given float values.

**Syntax rules for array literals:**
- Exactly one space must follow `[`
- Exactly one space must separate consecutive values
- Exactly one space must precede `]`
- Any violation (missing space, double space, non-float token) is a fatal error

To create a matrix, use `r` (reshape) after creating a flat tensor.

---

### Operations

Stack notation: leftmost item is TOS (top of stack), rightmost is bottom.

| Token | Name | Stack effect | Description |
|-------|------|--------------|-------------|
| `p` | print | `(a -- )` | Print tensor as flat array, then pop |
| `P` | print matrix | `(a -- )` | Print tensor row-by-row, then pop |
| `d` | duplicate | `(a -- a a)` | Duplicate top of stack (no copy, increments ref count) |
| `+` | add | `(a b -- a+b)` | Element-wise addition |
| `-` | subtract | `(a b -- a-b)` | Element-wise subtraction (TOS minus second) |
| `*` | multiply | `(a b -- a*b)` | Element-wise multiplication |
| `>` | greater | `(a b -- a>b)` | Element-wise comparison, result is boolean tensor |
| `<` | less | `(a b -- a<b)` | Element-wise comparison, result is boolean tensor |
| `=` | equal | `(a b -- a==b)` | Element-wise equality, result is boolean tensor |
| `&` | and | `(a b -- a&&b)` | Element-wise logical AND (inputs must be boolean) |
| `\|` | or | `(a b -- a\|\|b)` | Element-wise logical OR (inputs must be boolean) |
| `!` | not | `(a -- !a)` | Element-wise logical NOT (input must be boolean) |
| `$` | mask | `(m a b -- r)` | Element-wise conditional: `r[i] = m[i] ? a[i] : b[i]` (m must be boolean) |
| `@` | matmul | `(a b -- a@b)` | Matrix multiplication; both operands must be 2D (rows ≥ 2) |
| `r` | reshape | `(shape a -- a')` | Reshape `a` to dimensions given by `shape` (a `[1×1]` or `[1×2]` tensor) |
| `S` | sum | `(a -- s)` | Sum all elements, push result as `[1×1]` tensor |
| `.` | dot product | `(a b -- d)` | Dot product of two row vectors, push result as `[1×1]` tensor |
| `"…"` | push string | `( -- s)` | Push the filename literal `…` onto the stack as a string item |
| `(` | load image | `(s -- a)` | Pop filename string, read PGM P5 greyscale image, push tensor with values in `[0, 1]` |
| `)` | save image | `(a s -- )` | Pop filename string and tensor, write PGM P5 file; values clamped to `[0, 1]` then scaled to `[0, 255]` |
| `?` | random | `(shape -- a)` | Pop shape tensor (`[1×1]` or `[1×2]`), push tensor filled with uniform random floats in `[0, 1)` |
| `R` | relu | `(a -- relu(a))` | Element-wise ReLU: `max(0, x)` for each element |
| `m` | min | `(a b -- min(a,b))` | Element-wise minimum of two tensors |
| `M` | max | `(a b -- max(a,b))` | Element-wise maximum of two tensors |
| `s` | switch | `(a b -- b a)` | Swap the top two stack items (works for tensors and strings) |
| `o` | over | `(a b -- b a b)` | Copy the second-from-top item to the top of the stack |
| `D` | drop | `(a -- )` | Discard and free the top stack item |
| `_` | ravel | `(a -- a')` | Flatten tensor to a 1D row vector: `[r×c] → [1×(r*c)]` |
| `#` | shape | `(a -- s)` | Pop tensor, push its shape: `[n]` for 1D (1×1 tensor), `[rows cols]` for 2D (1×2 tensor) |
| `f` | fill | `(v s -- r)` | Pop shape `s` and value tensor `v`, push new tensor of shape `s` filled by cycling `v`'s elements |
| `}` | save disk | `(a f -- )` | Pop filename string `f` and tensor `a`, serialise to binary file (disk format) |
| `{` | load disk | `(f -- a)` | Pop filename string `f`, map file into memory with `mmap` (no copy), push tensor |
| `c` | convolve | `(a k -- r)` | 2D convolution of matrix `a` with kernel `k` (square, odd order), stride 1, zero-padding; result has same shape as `a` |

---

### Separator rules

Exactly one whitespace character must separate consecutive tokens. Leading and trailing whitespace is allowed; double spaces or missing spaces between tokens are fatal errors.

---

### Shape rules

- Binary element-wise operations (`+`, `-`, `*`, `>`, `<`, `=`, `&`, `|`, `$`, `m`, `M`) require both operands to have identical shapes.
- `f` cycles through `v` if `s` requests more elements than `v` contains.
- `@` requires `a.cols == b.rows` and both operands must be proper matrices (rows ≥ 2).
- `r` requires the total number of elements to be preserved: `new_rows * new_cols == old_rows * old_cols`.
- The `shape` operand for `r` is a `[1×1]` tensor (sets columns, keeps rows=1) or a `[1×2]` tensor `[rows cols]`.
- Boolean operations (`&`, `|`, `!`, `$`) require all elements to be exactly `0.0` or `1.0`.
- `?` consumes a shape tensor with the same convention as `r`: `[n]` → `[1×n]`, `[rows cols]` → `[rows×cols]`.

---

### Examples

**Create and print a vector:**
```
[ 1.0 2.0 3.0 4.0 ] p
```

**Element-wise operations:**
```
[ 1.0 2.0 3.0 ] [ 4.0 5.0 6.0 ] + p
```

**Reshape and matrix multiply:**
```
[ 1.0 2.0 3.0 4.0 5.0 6.0 ] [ 2.0 3.0 ] r
[ 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 ] [ 3.0 3.0 ] r
@ P
```

**Boolean masking (select elements from two tensors):**
```
[ 40.0 50.0 60.0 ] [ 10.0 20.0 30.0 ] [ 1.0 0.0 1.0 ] $ p
```
Result: `[10.0, 50.0, 30.0]` — where mask=1, takes from `a`; where mask=0, takes from `b`.

**Sum reduction:**
```
[ 1.0 2.0 3.0 4.0 5.0 ] S p
```

**Generate a random 3×4 matrix and save as PGM:**
```
[ 3.0 4.0 ] ? "output.pgm" )
```

**ReLU on a vector:**
```
[ -2.0 -1.0 0.0 1.0 2.0 ] R p
```

**Element-wise min/max:**
```
[ 1.0 5.0 3.0 ] [ 4.0 2.0 6.0 ] m p
[ 1.0 5.0 3.0 ] [ 4.0 2.0 6.0 ] M p
```

**Stack manipulation:**
```
[ 1.0 2.0 ] [ 3.0 4.0 ] s p p   # swap: prints [3 4] then [1 2]
[ 1.0 2.0 ] [ 3.0 4.0 ] o p     # over: copies second to top
[ 1.0 2.0 3.0 ] D p              # drop: discards [1 2 3], prints nothing new
```

**Shape inspection and ravel:**
```
[ 1.0 2.0 3.0 4.0 5.0 6.0 ] [ 2.0 3.0 ] r # p   # prints shape [2 3]
[ 1.0 2.0 3.0 4.0 5.0 6.0 ] [ 2.0 3.0 ] r _ p   # ravel to [1×6]
```

**Fill a tensor with a repeating pattern:**
```
[ 1.0 0.0 ] [ 4.0 4.0 ] f p   # 4×4 matrix alternating 1 and 0
```

**Save and reload a tensor from disk:**
```
[ 1.0 2.0 3.0 ] "out.bin" }
"out.bin" { p
```

**Image blur pipeline (reads a PGM, applies 5×5 box blur, writes result):**
```
"examples/cray-2.pgm" ( [ 5.0 5.0 ] [ 0.04 ] f c "examples/cray-2-blurred.pgm" )
```

Step-by-step:
1. `"examples/cray-2.pgm" (` — read PGM P5 greyscale image; pixel values normalised to `[0, 1]`
2. `[ 5.0 5.0 ] [ 0.04 ] f` — build 5×5 uniform blur kernel (each weight = 0.04, sum = 1.0)
3. `c` — apply 2D convolution with zero-padding; output same shape as input
4. `"examples/cray-2-blurred.pgm" )` — write result as PGM P5 (values clamped to `[0, 1]` → `[0, 255]`)

---

## Implementation Notes

### Memory management

Stack values are heap-allocated `array_instance` structs with a `ref_count` field. The `d` (duplicate) opcode increments the reference count instead of copying data; `instance_free` decrements it and frees when the count reaches zero. This avoids unnecessary allocations for operations like `d ... p`.

### Matrix multiplication

The `@` operator transposes the right-hand operand (`b`) for cache-friendly sequential access, then dispatches based on size. If all three dimensions fit within `BLOCK_SIZE` (default 64), a simple triple loop is used directly with no boundary-check overhead. Otherwise a blocked algorithm (64×64×64 tiles) is used, with the outer loops parallelized via OpenMP (`#pragma omp parallel for collapse(2)`).

### mmap load (`{`)

`{` maps the binary tensor file directly into the process address space with `mmap(PROT_READ, MAP_PRIVATE)` — no `malloc` or `fread` for the data. The `array_instance` holds a pointer into the mapped region; `instance_free` detects `on_disk == 1` and calls `munmap` instead of `free`, using `data_offset` stored in the instance to recover the original map base pointer.

### 2D Convolution (`c`)

`c` requires a square kernel of odd order (3×3, 5×5, …). It first allocates a zero-padded copy of the input (`padding`), then computes each output element as the dot product of the kernel with the corresponding window (`c_dot`). Output is written to a freshly allocated buffer — never in-place — so the operator is safe on mmap-backed tensors.

### Stack resizing

The stack doubles capacity when full and halves when usage drops below 25% of capacity. The 25% threshold avoids thrashing on push/pop sequences near the resize boundary.

### Layout

All tensors use row-major (C-order) layout. Element `[i, j]` of a tensor with `N` columns is at `data[i * N + j]`.

### Error reporting

All errors are printed to `stderr` in English. Each error message is emitted by the function that first detects the problem (which has the relevant context — shapes, values, filenames). `main` then prints a qualitative category line:

```
error: shape mismatch [1 2] != [1 3]
abort: shape mismatch
```

Error categories (returned as `TFError` codes from all ops): `stack error`, `shape mismatch`, `type error`, `invalid argument`, `I/O error`, `memory error`, `syntax error`. Exit code is always `1` on any failure.

### Array literal parsing

`parse_array` uses two passes: the first validates strict syntax and counts elements; the second allocates a single block and fills it. This avoids `realloc` churn and ensures memory is allocated exactly once per literal.

### Binary tensor format

The disk format is used by `{` (load) and `}` (save). Files start with a `disk_header` of fixed size, followed by float data at a fixed offset of 64 bytes:

```
int32_t shape[MAX_DIM]   // shape[0]=rows, shape[1]=cols
int32_t ndim             // 1 or 2
<4 bytes padding>        // compiler-inserted for off_t alignment
off_t   data_offset      // byte offset where float data begins (always 64)
```

Float data starts at `data_offset` (byte 64) and contains `rows * cols` IEEE 754 single-precision values in row-major order. If `ndim == 1`, the tensor is treated as a row vector `[1×cols]`. The 64-byte offset leaves room for future header extensions without breaking existing files.

---

## Project Structure

```
.
├── main.c                    # Entry point; reads script and runs interpreter
├── parser.c / parser.h       # Opcode dispatch table, token lookup, array/string parsing
├── ops_stack.c               # Stack operations: reshape, print, dup, switch, over, drop, ravel, shape, fill
├── ops_math.c                # Element-wise ops: arithmetic, comparisons, logical, relu, min/max
├── ops_linalg.c              # Linear algebra: matrix multiply (blocked), sum, dot product
├── ops_conv.c                # 2D convolution with zero-padding
├── ops_io.c                  # I/O: PGM load/save, binary disk format, random array
├── stack.c / stack.h         # Dynamic stack with reference-counted tensor values
├── reader.c / reader.h       # File I/O utilities
├── Makefile
├── run_tests.sh              # Test suite (82 tests, supports --valgrind)
└── examples/
    ├── cray-2.pgm                   # Sample greyscale image (PGM P5)
    ├── image_blur.tensorforth       # 5×5 box blur on cray-2.pgm
    ├── detect_edges.tensorforth     # Laplacian edge detection on cray-2.pgm
    ├── convert_to_bw.tensorforth    # Threshold to black & white
    ├── random_matmul.tensorforth    # Random matrix multiplication example
    ├── save_tensor.tensorforth      # Save/load roundtrip via binary disk format
    ├── duplicate.tensorforth        # Stack duplication example
    └── game_of_life.tensorforth     # Conway's Game of Life step (1000×1000)
```
