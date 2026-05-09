# TensorForth Interpreter

A stack-based interpreter for tensor computation, written in C17. TensorForth is a Forth-inspired language specialized for linear algebra operations on 2D tensors (matrices).

## Overview

TensorForth programs are sequences of single-character opcodes that operate on a stack of tensors. All values on the stack are 2D arrays (1D arrays are represented as row vectors with `shape = [1, N]`). Operations consume their operands from the stack and push results back.

## Building

Requires `gcc` with C17 support and `OpenMP`.

```sh
make        # debug build (-O0 -g)
make CFLAGS="-O3 -fopenmp" all   # optimized build
make clean  # remove build artifacts
```

This produces a `TensorForth` executable.

## Usage

```sh
./TensorForth <script.tensorforth>
```

## Language Reference

### Tensor Literal

```
[ v1 v2 ... vN ]
```

Pushes a `[1 × N]` tensor onto the stack with the given float values.

To create a matrix, use `r` (reshape) after creating a flat tensor.

---

### Operations

| Token | Name | Stack effect | Description |
|-------|------|--------------|-------------|
| `p` | print | `(a -- )` | Print tensor as flat array, then pop |
| `P` | print matrix | `(a -- )` | Print tensor row-by-row, then pop |
| `d` | duplicate | `(a -- a a)` | Duplicate top of stack (no copy, increments ref count) |
| `+` | add | `(a b -- a+b)` | Element-wise addition |
| `-` | subtract | `(a b -- a-b)` | Element-wise subtraction |
| `*` | multiply | `(a b -- a*b)` | Element-wise multiplication |
| `>` | greater | `(a b -- a>b)` | Element-wise comparison, result is boolean tensor |
| `<` | less | `(a b -- a<b)` | Element-wise comparison, result is boolean tensor |
| `=` | equal | `(a b -- a==b)` | Element-wise equality, result is boolean tensor |
| `&` | and | `(a b -- a&&b)` | Element-wise logical AND (inputs must be boolean) |
| `\|` | or | `(a b -- a\|\|b)` | Element-wise logical OR (inputs must be boolean) |
| `!` | not | `(a -- !a)` | Element-wise logical NOT (input must be boolean) |
| `$` | mask | `(m a b -- r)` | Element-wise conditional: `r[i] = m[i] ? a[i] : b[i]` |
| `@` | matmul | `(a b -- a@b)` | Matrix multiplication |
| `r` | reshape | `(shape a -- a')` | Reshape `a` to dimensions given by `shape` (a `[1×2]` tensor) |
| `S` | sum | `(a -- s)` | Sum all elements, push result as `[1×1]` tensor |
| `.` | dot product | `(a b -- d)` | Dot product of two flat tensors, push result as `[1×1]` tensor |

---

### Shape rules

- Binary element-wise operations (`+`, `-`, `*`, `>`, `<`, `=`, `&`, `|`, `$`) require both operands to have identical shapes.
- `@` requires `a.cols == b.rows`.
- `r` requires the total number of elements to be preserved: `new_rows * new_cols == old_rows * old_cols`.
- The `shape` operand for `r` must be a `[1×2]` tensor where `shape[0]` is the new row count and `shape[1]` is the new column count.

---

### Examples

**Create and print a vector:**
```
[ 1 2 3 4 ] p
```

**Element-wise operations:**
```
[ 1 2 3 ] [ 4 5 6 ] + p
```

**Reshape and matrix multiply:**
```
[ 1 2 3 4 5 6 ] [ 2 3 ] r
[ 1 0 0 0 1 0 0 0 1 ] [ 3 3 ] r
@ P
```

**Boolean masking:**
```
[ 1 -2 3 -4 ] d [ 0 1 0 1 ] r
[ 0 0 0 0 ] $
p
```

**Sum reduction:**
```
[ 1 2 3 4 5 ] S p
```

---

## Implementation Notes

### Memory management

Stack values are heap-allocated `array_instance` structs with a `ref_count` field. The `d` (duplicate) opcode increments the reference count instead of copying data; `stack_pop` decrements it and frees when the count reaches zero. This avoids unnecessary allocations for operations like `d ... p`.

### Matrix multiplication

The `@` operator uses a blocked algorithm (16 × 16 × 16 tiles) with an explicit transpose of the right-hand operand to ensure sequential memory access. The outer loops are parallelized with OpenMP (`#pragma omp parallel for collapse(2)`).

### Layout

All tensors use row-major (C-order) layout. Element `[i, j]` of a tensor with `N` columns is at `data[i * N + j]`.

---

## Project Structure

```
.
├── main.c                    # Entry point; reads script and runs interpreter
├── parser.c / parser.h       # Opcode dispatch and all operation implementations
├── stack.c / stack.h         # Dynamic stack with reference-counted tensor values
├── reader.c / reader.h       # File I/O utilities
├── Makefile
├── example.tensorforth
└── random_matmul.tensorforth
```
