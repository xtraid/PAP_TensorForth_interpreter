#include "parser.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Pops the top two tensors, applies element-wise op ('a'=add, 's'=subtract, 'p'=product),
 * and pushes the result. Handles aliased operands (e.g. after duplicate) safely.
 * Input: my_stack — the stack, op — operation code.
 * Output: TF_OK on success. */
int algebra (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return TF_ERR_STACK;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return TF_ERR_STACK; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "error: shape mismatch [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(a);
		/* instance_free uses ref_count: safe to call twice on aliased operands (2→1→0) */
		instance_free(b);
		return TF_ERR_SHAPE;
	}
	// perf: hoisted a->shape.row, a->shape.col, a->data, b->data — avoids reload each iteration
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	// perf: restrict on new_data — guarantees no aliasing with a_data/b_data, enables auto-vectorisation
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); instance_free(b); return TF_ERR_MEM; }
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'a'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] + b_data[i * cols + j];
			}
		}
	} else if(op == 's'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] - b_data[i * cols + j];
			}
		}
	} else if(op == 'p'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] * b_data[i * cols + j];
			}
		}
	}
	shape_t shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return TF_ERR_MEM; }
	return TF_OK;
}

/* Pops the top two tensors and applies element-wise comparison op ('M'=greater, 'm'=lesser, 'u'=equal).
 * Result elements are 1.0 (true) or 0.0 (false). Tensors must have identical shape.
 * Input: my_stack — the stack, op — comparison code.
 * Output: TF_OK on success. */
int comparisons (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return TF_ERR_STACK;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return TF_ERR_STACK; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "error: shape mismatch [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(a);
		instance_free(b);
		return TF_ERR_SHAPE;
	}
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); instance_free(b); return TF_ERR_MEM; }
	if (op == 'M'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] > b_data[i * cols + j];
			}
		}
	} else if(op == 'm'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] < b_data[i * cols + j];
			}
		}
	} else if(op == 'u'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] == b_data[i * cols + j];
			}
		}
	}
	shape_t shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return TF_ERR_MEM; }
	return TF_OK;
}

/* Pops the top two tensors and applies element-wise logical op ('a'=AND, 'o'=OR).
 * Operands must be boolean (all elements 0.0 or 1.0) and have identical shape.
 * Input: my_stack — the stack, op — logical operation code.
 * Output: TF_OK on success. */
int logical_binary (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return TF_ERR_STACK;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return TF_ERR_STACK; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "error: shape mismatch [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(a);
		instance_free(b);
		return TF_ERR_SHAPE;
	}
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	int n = rows * cols;
	float * restrict new_data = malloc(sizeof(float) * (size_t)n);
	if (new_data == NULL) { instance_free(a); instance_free(b); return TF_ERR_MEM; }
	for (int k = 0; k < n; k++) {
		if (a_data[k] != 0.f && a_data[k] != 1.f) {
			fprintf(stderr, "error: non-boolean value in first operand at [%d, %d] = %f\n", k / cols, k % cols, a_data[k]);
			free(new_data); instance_free(a); instance_free(b);
			return TF_ERR_TYPE;
		}
		if (b_data[k] != 0.f && b_data[k] != 1.f) {
			fprintf(stderr, "error: non-boolean value in second operand at [%d, %d] = %f\n", k / cols, k % cols, b_data[k]);
			free(new_data); instance_free(a); instance_free(b);
			return TF_ERR_TYPE;
		}
	}
	#pragma omp parallel for schedule(static)
	for (int k = 0; k < n; k++)
		new_data[k] = (op == 'a') ? (float)(a_data[k] && b_data[k]) : (float)(a_data[k] || b_data[k]);
	shape_t shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return TF_ERR_MEM; }
	return TF_OK;
}

/* Pops the top tensor and applies element-wise logical NOT (0->1, 1->0).
 * Fails if any element is not 0.0 or 1.0.
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int op_not (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return TF_ERR_STACK;
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); return TF_ERR_MEM; }
	int n_not = rows * cols;
	for (int i = 0; i < n_not; i++) {
		if (a_data[i] != 0.f && a_data[i] != 1.f) {
			fprintf(stderr, "error: '!' requires boolean array, found %f at (%d, %d)\n", a_data[i], i / cols, i % cols);
			free(new_data); instance_free(a); return TF_ERR_TYPE;
		}
	}
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_not; i++)
		new_data[i] = 1.f - a_data[i];
	shape_t shape;
	shape.row = rows;
	shape.col = cols;
	instance_free(a);
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return TF_ERR_MEM; }
	return TF_OK;
}

/* Pops m (mask), a, b from the stack; computes result[i] = m[i] ? a[i] : b[i].
 * m must be boolean (0.0/1.0); all three tensors must have identical shape.
 * Input: my_stack — the stack (top: m, then a, then b).
 * Output: TF_OK on success. */
int mask (stack *my_stack) {
	array_instance *m = stack_pop(my_stack);
	if (m == NULL) return TF_ERR_STACK;
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) { instance_free(m); return TF_ERR_STACK; }
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(m); instance_free(a); return TF_ERR_STACK; }

	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "error: shape mismatch a[%d %d] != b[%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(m); instance_free(a); instance_free(b);
		return TF_ERR_SHAPE;
	}
	if (a->shape.row != m->shape.row || a->shape.col != m->shape.col){
		fprintf(stderr, "error: shape mismatch a[%d %d] != mask[%d %d]\n",
			a->shape.row, a->shape.col, m->shape.row, m->shape.col);
		instance_free(m); instance_free(a); instance_free(b);
		return TF_ERR_SHAPE;
	}

	int m_rows = m->shape.row;
	int m_cols = m->shape.col;
	for (int i = 0; i < m_rows; i++){
		for (int j = 0; j < m_cols; j++){
			if (m->data[i * m_cols + j] != 0.f && m->data[i * m_cols + j] != 1.f){
				fprintf(stderr, "error: '$' requires boolean mask, found %f at (%d, %d)\n",
					m->data[i * m_cols + j], i, j);
				instance_free(m); instance_free(a); instance_free(b);
				return TF_ERR_TYPE;
			}
		}
	}
	float *new_data = malloc(sizeof(float) * (size_t)(m_rows * m_cols));
	if (new_data == NULL) { instance_free(m); instance_free(a); instance_free(b); return TF_ERR_MEM; }
	/* ternary generates cmov with -O3, branchless and correct on inf/NaN */
	int nm = m_rows * m_cols;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < nm; i++)
		new_data[i] = (m->data[i] != 0.0f) ? a->data[i] : b->data[i];

	shape_t shape = {m_rows, m_cols};
	int push_err = stack_push(my_stack, new_data, shape);
	if (push_err != 0) free(new_data);
	instance_free(m);
	instance_free(a);
	instance_free(b);
	return push_err != 0 ? TF_ERR_MEM : TF_OK;
}

/* Pops two tensors a (top) and b of identical shape, pushes element-wise min or max.
 * Input: my_stack — the stack, op — 'm' for min, 'M' for max.
 * Output: TF_OK on success. */
int extrema(stack *my_stack, char op){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return TF_ERR_STACK;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return TF_ERR_STACK; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "error: shape mismatch [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(a);
		instance_free(b);
		return TF_ERR_SHAPE;
	}
	int row = a->shape.row;
	int col = a->shape.col;
	int n = row * col;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); instance_free(b); return TF_ERR_MEM; }
	/* fminf/fmaxf are branchless with -O3 and correct on inf/NaN */
	if (op == 'm'){
		for (int i = 0; i < n; i++)
			new_data[i] = fminf(a->data[i], b->data[i]);
	} else {
		for (int i = 0; i < n; i++)
			new_data[i] = fmaxf(a->data[i], b->data[i]);
	}
	instance_free(a);
	instance_free(b);
	shape_t shape = {row, col};
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return TF_ERR_MEM; }
	return TF_OK;
}

/* Pops a tensor and pushes a new tensor with ReLU applied element-wise: max(0, x).
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int relu (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (!a) return TF_ERR_STACK;
	int n = a->shape.row * a->shape.col;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); return TF_ERR_MEM; }
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		new_data[i] = (a->data[i] > 0.f) ? a->data[i] : 0.f;
	shape_t shape = {a->shape.row, a->shape.col};
	instance_free(a);
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return TF_ERR_MEM; }
	return TF_OK;
}
