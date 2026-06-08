#include "parser.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>

/* Pops s (shape tensor: 1D, 1 or 2 elements) then a, reshapes a to the dimensions in s.
 * Memory layout is unchanged; only shape.row and shape.col are updated.
 * Validates dimensions >= 1 and product <= INT_MAX (entry point for user-supplied dims).
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int op_reshape(stack *my_stack){
	array_instance *s = stack_pop(my_stack);
	if (s == NULL) return TF_ERR_STACK;
	if (s->shape.row != 1 || s->shape.col < 1 || s->shape.col > 2){
		fprintf(stderr, "error: 'r' requires a 1D shape tensor with 1 or 2 elements\n");
		instance_free(s);
		return TF_ERR_ARG;
	}
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) { instance_free(s); return TF_ERR_STACK; }

	int new_rows = (s->shape.col == 2) ? (int)s->data[0] : 1;
	int new_cols = (s->shape.col == 2) ? (int)s->data[1] : (int)s->data[0];

	if (new_rows < 1 || new_cols < 1) {
		fprintf(stderr, "error: 'r' requires dimensions >= 1\n");
		instance_free(s); instance_free(a); return TF_ERR_ARG;
	}
	int64_t n64 = (int64_t)new_rows * (int64_t)new_cols;
	if (n64 > INT_MAX) {
		fprintf(stderr, "error: tensor too large\n");
		instance_free(s); instance_free(a); return TF_ERR_ARG;
	}
	if (new_rows * new_cols != a->shape.row * a->shape.col){
		fprintf(stderr, "error: reshape incompatible [%d %d] -> [%d %d]\n",
			a->shape.row, a->shape.col, new_rows, new_cols);
		instance_free(s);
		instance_free(a);
		return TF_ERR_SHAPE;
	}
	instance_free(s);
	int n = new_rows * new_cols;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); return TF_ERR_MEM; }
	memcpy(new_data, a->data, sizeof(float) * (size_t)n);
	instance_free(a);
	shape_t shape = {new_rows, new_cols};
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return TF_ERR_MEM; }
	return TF_OK;
}



/* Pops the top tensor and prints it in the format: Tensor(shape=[r c], data=[...]).
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int pop_print(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	if (last == NULL) return TF_ERR_STACK;
	// perf: hoisted last->shape.row, last->shape.col, last->data — avoids repeated struct dereference in loop
	int rows      = last->shape.row;
	int cols      = last->shape.col;
	float *data   = last->data;
	int total = rows * cols;
	printf("Tensor(shape=[%d %d], data=[", rows, cols);
	for (int i = 0; i < total; i++){
		printf("%f", data[i]);
		if (i < total - 1) printf(" ");
	}
	printf("])\n");
	instance_free(last);
	return TF_OK;
}

/* Pops the top tensor and prints it row by row for visual debugging.
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int pop_print_as_matrix(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	if (last == NULL) return TF_ERR_STACK;
	printf("on top of the stack was\n");
	// perf: hoisted last->shape.row, last->shape.col, last->data — avoids reload each iteration
	int rows    = last->shape.row;
	int cols    = last->shape.col;
	float *data = last->data;
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	for (int i = 0; i < rows; i++){
		printf("[ ");
		for (int j = 0; j < cols; j++){
			printf("%f ", data[i * cols + j]);
		}
		printf("]\n");
	}
	instance_free(last);
	return TF_OK;
}



/* Duplicates the top element of the stack by incrementing its ref_count.
 * Does not allocate a new tensor — both stack entries point to the same instance.
 * Input: my_stack — the stack. */
int duplicate (stack *my_stack){
	array_instance *top = stack_peek(my_stack);
	if (top == NULL) return TF_ERR_STACK;
	return stack_push_instance(my_stack, top) != 0 ? TF_ERR_MEM : TF_OK;
}

/* Swaps the top two stack items. Works for both tensors and strings.
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int op_switch(stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return TF_ERR_STACK;
	stack_item b = stack_pop_item(my_stack);
	if (b.type == ITEM_NONE) {
		stack_free_item(a);
		return TF_ERR_STACK;
	}
	int err = stack_push_item(my_stack, a);
	stack_free_item(a);
	if (err != 0) { stack_free_item(b); return TF_ERR_MEM; }
	err = stack_push_item(my_stack, b);
	stack_free_item(b);
	return err != 0 ? TF_ERR_MEM : TF_OK;
}

/* Copies the second-from-top item to the top: ( a b -- a b a ).
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int over (stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return TF_ERR_STACK;
	stack_item b = stack_pop_item(my_stack);
	if (b.type == ITEM_NONE) {
		stack_free_item(a);
		return TF_ERR_STACK;
	}
	int err = stack_push_item(my_stack, b);
	if (err != 0) {
		stack_free_item(b);
		stack_free_item(a);
		return TF_ERR_MEM;
	}
	err = stack_push_item(my_stack, a);
	stack_free_item(a);
	if (err != 0) {
		stack_free_item(b);
		return TF_ERR_MEM;
	}
	err = stack_push_item(my_stack, b);
	stack_free_item(b);
	return err != 0 ? TF_ERR_MEM : TF_OK;
}

/* Pops and discards the top stack item.
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int drop (stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return TF_ERR_STACK;
	stack_free_item(a);
	return TF_OK;
}

/* Flattens the top tensor to a 1D row vector in place: shape [r c] -> [1, r*c].
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int ravel(stack *my_stack) {
	array_instance *a = stack_pop(my_stack);
	if (!a) return TF_ERR_STACK;
	if (a->ref_count == 1){
		a->shape.col = a->shape.row * a->shape.col;
		a->shape.row = 1;
		if (stack_push_instance(my_stack, a) != 0) {
			instance_free(a);
			return TF_ERR_MEM;
		}
		instance_free(a);
	} else {
		int new_col = a->shape.row * a->shape.col;
		float *new_data = malloc(sizeof(float) * (size_t)new_col);
		if (!new_data) { instance_free(a); return TF_ERR_MEM; }
		memcpy(new_data, a->data, (size_t)new_col * sizeof(float));

		shape_t shape = {1, new_col};
		instance_free(a);
		if (stack_push(my_stack, new_data, shape) != 0) {
			fprintf(stderr, "error: '_': stack push failed\n");
			free(new_data);
			return TF_ERR_MEM;
		}
	}
	return TF_OK;
}

/* Pops a tensor and pushes a 1D tensor containing its shape.
 * 2D tensor [r c] -> [r c] (1x2); 1D tensor [1 n] -> [n] (1x1).
 * Input: my_stack — the stack.
 * Output: TF_OK on success. */
int op_shape(stack *my_stack) {
	array_instance *a = stack_pop(my_stack);
	if (!a) return TF_ERR_STACK;
	int is_1d = (a->shape.row == 1);
	int count = is_1d ? 1 : 2;
	float *shape_arr = malloc(sizeof(float) * (size_t)count);
	if (!shape_arr) {
		instance_free(a);
		return TF_ERR_MEM;
	}
	if (is_1d) {
		shape_arr[0] = (float)a->shape.col;
	} else {
		shape_arr[0] = (float)a->shape.row;
		shape_arr[1] = (float)a->shape.col;
	}
	instance_free(a);
	shape_t shape = {1, count};
	if (stack_push(my_stack, shape_arr, shape) != 0) {
		free(shape_arr);
		return TF_ERR_MEM;
	}
	return TF_OK;
}


/* Pops a value tensor v (top) and a shape tensor s, pushes a new tensor of shape s
 * filled by cycling through the elements of v.
 * Validates dimensions >= 1 and product <= INT_MAX (entry point for user-supplied dims).
 * Input: my_stack — the stack (top: v, then s).
 * Output: TF_OK on success. */
int fill(stack *my_stack) {
	array_instance *v = stack_pop(my_stack);
	if (!v) return TF_ERR_STACK;
	array_instance *s = stack_pop(my_stack);
	if (!s) { instance_free(v); return TF_ERR_STACK; }

	if (s->shape.row != 1 || s->shape.col < 1 || s->shape.col > 2) {
		fprintf(stderr, "error: 'f' requires a 1D shape tensor with 1 or 2 elements\n");
		instance_free(s); instance_free(v); return TF_ERR_ARG;
	}
	int row = (s->shape.col == 2) ? (int)s->data[0] : 1;
	int col = (s->shape.col == 2) ? (int)s->data[1] : (int)s->data[0];
	if (row < 1 || col < 1) {
		fprintf(stderr, "error: 'f' requires dimensions >= 1\n");
		instance_free(s); instance_free(v); return TF_ERR_ARG;
	}
	instance_free(s);

	int64_t n64 = (int64_t)row * (int64_t)col;
	if (n64 > INT_MAX) {
		fprintf(stderr, "error: tensor too large\n");
		instance_free(v); return TF_ERR_ARG;
	}
	int n = (int)n64;
	int m = v->shape.row * v->shape.col;
	if (m <= 0) {
		fprintf(stderr, "error: 'f': value tensor cannot be empty\n");
		instance_free(v);
		return TF_ERR_ARG;
	}
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(v); return TF_ERR_MEM; }

	for (int i = 0; i < n; i++)
		new_data[i] = v->data[i % m];

	instance_free(v);
	shape_t shape = {row, col};
	if (stack_push(my_stack, new_data, shape) != 0) {
		free(new_data);
		return TF_ERR_MEM;
	}
	return TF_OK;
}
