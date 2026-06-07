#include "parser.h"
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 64

/* Transposes a row-major matrix of shape [rows x cols] into new_data of shape [cols x rows].
 * new_data must be pre-allocated with rows*cols floats. */
static void transpose(float *data, float *new_data, int rows, int cols){
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			new_data[j * rows + i] = data[i * cols + j];
}

/* kernel e blocked_multiply ricevono B già trasposta (B_T shape [b_cols × a_cols]).
 * C[i][j] += A[i][k] * B_T[j][k]  =  C = A * B_originale
 * Entrambi gli accessi sono sequenziali in k -> cache friendly. */
static void kernel(float *A, float *B_T, float *C,
                   int x, int s1, int y, int s2, int k, int s3,
                   int a_rows, int a_cols, int b_cols){
    int mx = x + s1 < a_rows ? x + s1 : a_rows;
    int my = y + s2 < b_cols ? y + s2 : b_cols;
    int mk = k + s3 < a_cols ? k + s3 : a_cols;
    for (int i = x; i < mx; i++)
        for (int j = y; j < my; j++) {
            float sum = C[i * b_cols + j];
            for (int kk = k; kk < mk; kk++)
                sum += A[i * a_cols + kk] * B_T[j * a_cols + kk];
            C[i * b_cols + j] = sum;
        }
}

/* Multiplica A [a_rows × a_cols] per B_T [b_cols × a_cols] (già trasposta) e accumula in C [a_rows × b_cols].
 * Suddivide il lavoro in blocchi di dimensione s1×s2×s3 per migliorare la località di cache;
 * parallelizza i loop esterni con OpenMP. */
static void blocked_multiply(float *A, float *B_T, float *C, int a_rows, int a_cols, int b_cols){
	if (a_rows <= BLOCK_SIZE && a_cols <= BLOCK_SIZE && b_cols <= BLOCK_SIZE) {
		for (int i = 0; i < a_rows; i++)
			for (int j = 0; j < b_cols; j++) {
				float sum = 0.0f;
				for (int k = 0; k < a_cols; k++)
					sum += A[i * a_cols + k] * B_T[j * a_cols + k];
				C[i * b_cols + j] = sum;
			}
		return;
	}
	#pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < a_rows; i += BLOCK_SIZE)
		for (int j = 0; j < b_cols; j += BLOCK_SIZE)
			for (int k = 0; k < a_cols; k += BLOCK_SIZE)
				kernel(A, B_T, C, i, BLOCK_SIZE, j, BLOCK_SIZE, k, BLOCK_SIZE, a_rows, a_cols, b_cols);
}

/* Pops a (top) and b from the stack and pushes a@b (matrix product).
 * Requires a.cols == b.rows and rows > 1. Result shape is [a.rows x b.cols].
 * Transposes b internally for cache-friendly access, then uses blocked_multiply.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int mat_mat_mul(stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }

	int a_rows = a->shape.row;
	int a_cols = a->shape.col;
	int b_rows = b->shape.row;
	int b_cols = b->shape.col;

	if (a_rows < 2 || b_rows < 2) {
		fprintf(stderr, "errore: @ richiede matrici 2D, non vettori (a[%d %d] b[%d %d])\n",
			a_rows, a_cols, b_rows, b_cols);
		instance_free(a);
		instance_free(b);
		return -1;
	}

	if (a_cols != b_rows){
		fprintf(stderr, "errore: shape incompatibili per prodotto matriciale a[%d %d] b[%d %d]\n",
			a_rows, a_cols, b_rows, b_cols);
		instance_free(a);
		instance_free(b);
		return -1;
	}

	float *b_T = malloc(sizeof(float) * (size_t)(b_rows * b_cols));
	if (b_T == NULL) { instance_free(a); instance_free(b); return -1; }
	transpose(b->data, b_T, b_rows, b_cols);

	float *new_data = calloc((size_t)(a_rows * b_cols), sizeof(float));
	if (new_data == NULL) { free(b_T); instance_free(a); instance_free(b); return -1; }

	blocked_multiply(a->data, b_T, new_data, a_rows, a_cols, b_cols);

	free(b_T);
	instance_free(a);
	instance_free(b);
	coppia shape = {a_rows, b_cols};
	return stack_push(my_stack, new_data, shape);
}


/* Pops the top tensor and pushes a scalar [1×1] with the sum of all its elements.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int sum_arr (stack *my_stack){
	array_instance *arr = stack_pop(my_stack);
	if (arr == NULL) return -1;
	int n = arr->shape.row * arr->shape.col;
	float total = 0.0f;
	#pragma omp parallel for reduction(+:total) schedule(static)
	for (int i = 0; i < n; i++)
		total += arr->data[i];
	instance_free(arr);
	float *sum = malloc(sizeof(float));
	if (sum == NULL) return -1;
	sum[0] = total;

	coppia shape;
	shape.row = 1;
	shape.col = 1;
	if (stack_push(my_stack, sum, shape) != 0) { free(sum); return -1; }
	return 0;
}



/* Pops two 1D vectors (shape [1×n]) and pushes a scalar [1×1] with their dot product.
 * Both operands must be row vectors (shape.row == 1) of equal length.
 * Input: my_stack — the stack (top: a, then b).
 * Output: 0 on success, -1 on error. */
int dot_product (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) {
		instance_free (a);
		return -1;
	}

	if (a->shape.row > 1) {
		fprintf (stderr, " errore: il primo argomento non è un vettore");
		instance_free(a);
		instance_free(b);
		return -1;
	}
	if (b->shape.row > 1) {
		fprintf (stderr, " errore: il secondo argomento non è un vettore");
		instance_free(a);
        instance_free(b);
		return -1;
	}
	if (a->shape.col != b->shape.col) {
		fprintf(stderr, "errore: lunghezze incompatibili [%d] != [%d]\n", a->shape.col, b->shape.col);
		instance_free(a);
		instance_free(b);
		return -1;
	}
	int n = a->shape.col;
	float *sum = malloc(sizeof(float));
	if (sum == NULL) { instance_free(a); instance_free(b); return -1; }
	float dot = 0.0f;
	#pragma omp parallel for reduction(+:dot) schedule(static)
	for (int i = 0; i < n; i++)
		dot += a->data[i] * b->data[i];
	sum[0] = dot;

	instance_free(a);
	instance_free(b);

	coppia shape;
	shape.row = 1;
	shape.col = 1;
	if (stack_push(my_stack, sum, shape) != 0) { free(sum); return -1; }
	return 0;
}
