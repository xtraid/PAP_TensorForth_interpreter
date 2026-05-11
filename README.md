# TensorForth Interpreter

A stack-based interpreter for tensor computation, written in C17. TensorForth is a Forth-inspired language specialized for linear algebra operations on 2D tensors (matrices).

## Overview

TensorForth programs are sequences of single-character opcodes that operate on a stack of tensors. All values on the stack are 2D arrays (1D arrays are represented as row vectors with `shape = [1, N]`). Operations consume their operands from the stack and push results back.

## Building

Requires `gcc` with C17 support and `OpenMP`.

```sh
make              # optimized build (-O3)
make DEBUG=TRUE   # debug build (-O0 -g)
make clean        # remove build artifacts
```

This produces a `TensorForth` executable.

## Usage

```sh
./TensorForth <script.tensorforth>
./run_tests.sh    # run the test suite (40 tests)
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
| `(` | load tensor | `(s -- a)` | Pop filename string, read binary `.bin` file, push resulting tensor |

---

### Shape rules

- Binary element-wise operations (`+`, `-`, `*`, `>`, `<`, `=`, `&`, `|`, `$`) require both operands to have identical shapes.
- `@` requires `a.cols == b.rows` and both operands must be proper matrices (rows ≥ 2).
- `r` requires the total number of elements to be preserved: `new_rows * new_cols == old_rows * old_cols`.
- The `shape` operand for `r` is a `[1×1]` tensor (sets columns, keeps rows=1) or a `[1×2]` tensor `[rows cols]`.
- Boolean operations (`&`, `|`, `!`, `$`) require all elements to be exactly `0.0` or `1.0`.

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

**Load a tensor from a binary file:**
```
"tests/test_tensor.bin" ( p
```

---

## Implementation Notes

### Memory management

Stack values are heap-allocated `array_instance` structs with a `ref_count` field. The `d` (duplicate) opcode increments the reference count instead of copying data; `instance_free` decrements it and frees when the count reaches zero. This avoids unnecessary allocations for operations like `d ... p`.

### Matrix multiplication

The `@` operator transposes the right-hand operand (`b`) for cache-friendly sequential access, then uses a blocked algorithm (16×16×16 tiles). The outer loops are parallelized with OpenMP (`#pragma omp parallel for collapse(2)`).

### Stack resizing

The stack doubles capacity when full and halves when usage drops below 25% of capacity. The 25% threshold avoids thrashing on push/pop sequences near the resize boundary.

### Layout

All tensors use row-major (C-order) layout. Element `[i, j]` of a tensor with `N` columns is at `data[i * N + j]`.

### Array literal parsing

`parse_array` uses two passes: the first validates strict syntax and counts elements; the second allocates a single block and fills it. This avoids `realloc` churn and ensures memory is allocated exactly once per literal.

### Binary tensor format

Binary `.bin` files are read by `read_image`. The file starts with a fixed-size header:

```
int32_t shape[MAX_DIM]   // shape[0]=rows, shape[1]=cols
int32_t ndim             // 1 or 2
<4 bytes padding>        // compiler-inserted for off_t alignment
off_t   data_offset      // byte offset where float data begins
```

Float data starts at `data_offset` and contains `rows * cols` IEEE 754 single-precision values in row-major order. If `ndim == 1`, `col` is set to 1 and the tensor is treated as a column vector.

---

## Project Structure

```
.
├── main.c                    # Entry point; reads script and runs interpreter
├── parser.c / parser.h       # Opcode dispatch and all operation implementations
├── stack.c / stack.h         # Dynamic stack with reference-counted tensor values
├── reader.c / reader.h       # File I/O utilities
├── Makefile
├── run_tests.sh              # Test suite (40 tests)
├── examples/                 # Sample .tensorforth scripts
│   ├── example.tensorforth
│   ├── random_matmul.tensorforth
│   └── test_complex.tensorforth
└── tests/                    # Test data files
    ├── test_tensor.bin
    ├── test_load.tensorforth
    └── cray-2.pgm
```
