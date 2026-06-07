#include "parser.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Pops the top two tensors, applies element-wise op ('a'=add, 's'=subtract, 'p'=product),
 * and pushes the result. Handles aliased operands (e.g. after duplicate) safely.
 * Input: my_stack — the stack, op — operation code.
 * Output: 0 on success. */
int algebra (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
			instance_free(a);
			/* instance_free usa ref_count: chiamarlo due volte è sicuro anche se a==b (2→1→0) */
			instance_free(b);
			return -1;
	}
	// perf: hoisted a->shape.row, a->shape.col, a->data, b->data — avoids reload each iteration
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	// perf: restrict on new_data — guarantees no aliasing with a_data/b_data, enables auto-vectorisation
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); instance_free(b); return -1; }
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
	coppia shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}

/* Pops the top two tensors and applies element-wise comparison op ('M'=greater, 'm'=lesser, 'u'=equal).
 * Result elements are 1.0 (true) or 0.0 (false). Tensors must have identical shape.
 * Input: my_stack — the stack, op — comparison code.
 * Output: 0 on success, -1 on error. */
int disuguaglianze (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
			instance_free(a);
			instance_free(b);
			return -1;
	}
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); instance_free(b); return -1; }
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
	coppia shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}

/* Pops the top two tensors and applies element-wise logical op ('a'=AND, 'o'=OR).
 * Operands must be boolean (all elements 0.0 or 1.0) and have identical shape.
 * Input: my_stack — the stack, op — logical operation code.
 * Output: 0 on success, -1 on shape error, -2 on non-boolean element. */
int op_logiche_2_arg (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
			instance_free(a);
			instance_free(b);
			return -1;
	}
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	int n = rows * cols;
	float * restrict new_data = malloc(sizeof(float) * (size_t)n);
	if (new_data == NULL) { instance_free(a); instance_free(b); return -1; }
	for (int k = 0; k < n; k++) {
		if (a_data[k] != 0.f && a_data[k] != 1.f) {
			fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del primo tensore è %f\n", k / cols, k % cols, a_data[k]);
			free(new_data); instance_free(a); instance_free(b);
			return -2;
		}
		if (b_data[k] != 0.f && b_data[k] != 1.f) {
			fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del secondo tensore è %f\n", k / cols, k % cols, b_data[k]);
			free(new_data); instance_free(a); instance_free(b);
			return -2;
		}
	}
	#pragma omp parallel for schedule(static)
	for (int k = 0; k < n; k++)
		new_data[k] = (op == 'a') ? (float)(a_data[k] && b_data[k]) : (float)(a_data[k] || b_data[k]);
	coppia shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}

/* Pops the top tensor and applies element-wise logical NOT (0->1, 1->0).
 * Fails if any element is not 0.0 or 1.0.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_not (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); return -1; }
	int n_not = rows * cols;
	for (int i = 0; i < n_not; i++) {
		if (a_data[i] != 0.f && a_data[i] != 1.f) {
			fprintf(stderr, "errore: not su array non booleano, elemento (%d, %d) = %f\n", i / cols, i % cols, a_data[i]);
			free(new_data); instance_free(a); return -1;
		}
	}
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_not; i++)
		new_data[i] = 1.f - a_data[i];
	coppia shape;
	shape.row = rows;
	shape.col = cols;
	instance_free(a);
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}

/* Pops m (mask), a, b from the stack; computes result[i] = m[i] ? a[i] : b[i].
 * m must be boolean (0.0/1.0); all three tensors must have identical shape.
 * Input: my_stack — the stack (top: m, then a, then b).
 * Output: 0 on success, -1 on error. */
int mask (stack *my_stack) {
	array_instance *m = stack_pop(my_stack);
	if (m == NULL) return -1;
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) {
		instance_free(m); return -1;
		}
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) {
		instance_free(m);
		instance_free(a);
		return -1; }

	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili a[%d %d] != b[%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(m); instance_free(a); instance_free(b);
		return -1;
	}
	if (a->shape.row != m->shape.row || a->shape.col != m->shape.col){
		fprintf(stderr, "errore: shape incompatibili a[%d %d] != m[%d %d]\n",
			a->shape.row, a->shape.col, m->shape.row, m->shape.col);
		instance_free(m); instance_free(a); instance_free(b);
		return -1;
	}

	int m_rows = m->shape.row;
	int m_cols = m->shape.col;
	for (int i = 0; i < m_rows; i++){
		for (int j = 0; j < m_cols; j++){
			if (m->data[i * m_cols + j] != 0.f && m->data[i * m_cols + j] != 1.f){
				fprintf(stderr, "errore: mask su maschera non booleana, elemento (%d, %d) = %f\n", i, j, m->data[i * m_cols + j]);
				instance_free(m);
				instance_free(a);
				instance_free(b);
				return -1;

			}
		}
	}
	float *new_data = malloc(sizeof(float)*(size_t)(m_rows*m_cols));
	if (new_data == NULL) {
		instance_free(m);
		instance_free(a);
		instance_free(b);
		return -1;
		}
	/* Precedente: select aritmetico m*a + (1-m)*b.
	 * Rimosso perché inf*0 = NaN contamina il risultato anche nel ramo non scelto.
	 * Il ternario genera cmov con -O3, è branchless e corretto su inf/NaN. */
	int nm = m_rows * m_cols;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < nm; i++)
		new_data[i] = (m->data[i] != 0.0f) ? a->data[i] : b->data[i];

	coppia shape;
	shape.row = m_rows;
	shape.col = m_cols;
	int err = stack_push(my_stack, new_data, shape);
	instance_free(m);
	instance_free(a);
	instance_free(b);
	return err;
}

/* Pops two tensors a (top) and b of identical shape, pushes element-wise min or max.
 * Input: my_stack — the stack, op — 'm' for min, 'M' for max.
 * Output: 0 on success, -1 on error. */
int extrema(stack *my_stack, char op){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n", a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(a);
		instance_free(b);
		return -1;
	}
	int row = a->shape.row;
	int col = a->shape.col;
	int n = row * col;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); instance_free(b); return -1; }
	/* Precedente: select aritmetico branchless x*cond + y*(1-cond).
	 * Rimosso perché inf*0 = NaN contamina il risultato anche nel ramo non scelto.
	 * fminf/fmaxf sono già branchless con -O3 e corretti su inf/NaN. */
	if (op == 'm'){
		for (int i = 0; i < n; i++)
			new_data[i] = fminf(a->data[i], b->data[i]);
	} else {
		for (int i = 0; i < n; i++)
			new_data[i] = fmaxf(a->data[i], b->data[i]);
	}
	instance_free(a);
	instance_free(b);
	coppia shape = {row, col};
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}

/* Pops a tensor and pushes a new tensor with ReLU applied element-wise: max(0, x).
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int relu (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (!a) return -1;
	int n = a->shape.row * a->shape.col;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); return -1; }
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		new_data[i] = (a->data[i] > 0.f) ? a->data[i] : 0.f;
	coppia shape;
	shape.row = a->shape.row;
	shape.col = a->shape.col;
	int err = stack_push(my_stack, new_data,shape);
	if(err != 0){
		free(new_data);
		instance_free(a);
		return err;
	}
	instance_free(a);
	return 0;
}
