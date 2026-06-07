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
 * Output: 0 on success, -1 on error. */
int op_reshape(stack *my_stack){
	array_instance *s = stack_pop(my_stack);
	if (s == NULL) return -1;
	if (s->shape.row != 1 || s->shape.col < 1 || s->shape.col > 2){
		fprintf(stderr, "errore: reshape richiede un tensore 1D di 1 o 2 elementi\n");
		instance_free(s);
		return -1;
	}
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) { instance_free(s); return -1; }

	int new_rows = (s->shape.col == 2) ? (int)s->data[0] : 1;
	int new_cols = (s->shape.col == 2) ? (int)s->data[1] : (int)s->data[0];

	if (new_rows < 1 || new_cols < 1) {
		fprintf(stderr, "errore: reshape richiede dimensioni >= 1\n");
		instance_free(s); instance_free(a); return -1;
	}
	int64_t n64 = (int64_t)new_rows * (int64_t)new_cols;
	if (n64 > INT_MAX) {
		fprintf(stderr, "errore: tensore troppo grande\n");
		instance_free(s); instance_free(a); return -1;
	}
	if (new_rows * new_cols != a->shape.row * a->shape.col){
		fprintf(stderr, "errore: reshape incompatibile [%d %d] -> [%d %d]\n",
			a->shape.row, a->shape.col, new_rows, new_cols);
		instance_free(s);
		instance_free(a);
		return -1;
	}
	instance_free(s);
	int n = new_rows * new_cols;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); return -1; }
	memcpy(new_data, a->data, sizeof(float) * (size_t)n);
	instance_free(a);
	coppia shape = {new_rows, new_cols};
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}



/* Pops the top tensor and prints it in the format: Tensor(shape=[r c], data=[...]).
 * Input: my_stack — the stack.
 * Output: 0 on success. */
int pop_print(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	if (last == NULL) return -1;
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
	return 0;
}

/* Pops the top tensor and prints it row by row for visual debugging.
 * Input: my_stack — the stack.
 * Output: 0 on success. */
int pop_print_as_matrix(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	if (last == NULL) return -1;
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
	return 0;
}



/* Duplicates the top element of the stack by incrementing its ref_count.
 * Does not allocate a new tensor — both stack entries point to the same instance.
 * Input: my_stack — the stack. */
int duplicate (stack *my_stack){
	array_instance *top = stack_peek(my_stack);
	if (top == NULL) return -1;
	return stack_push_instance(my_stack, top);
}

/* Swaps the top two stack items. Works for both tensors and strings.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_switch(stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return -1;
	stack_item b = stack_pop_item(my_stack);
	if (b.type == ITEM_NONE) {
		stack_free_item(a);
		return -1;
	}
	int err = stack_push_item(my_stack, a);
	stack_free_item(a);
	if (err != 0) { stack_free_item(b); return -1; }
	err = stack_push_item(my_stack, b);
	stack_free_item(b);
	return err;
}

/* Copies the second-from-top item to the top: ( a b -- a b a ).
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int over (stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return -1;
	stack_item b = stack_pop_item(my_stack);
	if (b.type == ITEM_NONE) {
		stack_free_item(a);
		return -1;
	}
	int err = stack_push_item(my_stack, b);
	if (err != 0) {
		stack_free_item(b);
		stack_free_item(a);
		return -1;
	}
	err = stack_push_item(my_stack, a);
	stack_free_item(a);
	if (err != 0) {
		stack_free_item(b);
		return -1;
	}
	err = stack_push_item(my_stack, b);
	stack_free_item(b);
	return err;
}

/* Pops and discards the top stack item.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int drop (stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return -1;
	stack_free_item(a);
	return 0;
}

/* Flattens the top tensor to a 1D row vector in place: shape [r c] -> [1, r*c].
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int ravel(stack *my_stack) {
  array_instance *a = stack_pop(my_stack);
  if (!a) return -1;
  if (a->ref_count == 1){
    a->shape.col = a->shape.row * a->shape.col;
    a->shape.row = 1;
    if (stack_push_instance(my_stack, a) != 0) {
      instance_free(a);
      return -1;
    }
    instance_free(a);}
  else{
    int new_col = a->shape.row * a->shape.col;
    float *new_data = malloc(sizeof(float) * (size_t)new_col);
    memcpy(new_data, a->data, (size_t)new_col * sizeof(float));

    coppia shape;
    shape.row = 1;
    shape.col = new_col;

    instance_free(a);
    if (stack_push(my_stack, new_data, shape) != 0) {
      fprintf(stderr, "ravel: push_stack ha fallito");
      free(new_data);
      return -1;
    }
  }
	return 0;
}

/* Pops a tensor and pushes a 1D tensor [1×2] containing its shape [rows, cols].
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_shape(stack *my_stack) {
	array_instance *a = stack_pop(my_stack);
	if (!a) return -1;
	int is_1d = (a->shape.row == 1);
	int count = is_1d ? 1 : 2;
	float *shape_arr = malloc(sizeof(float) * (size_t)count);
	if (!shape_arr) {
		perror("malloc");
		instance_free(a);
		return -1;
	}
	if (is_1d) {
		shape_arr[0] = (float)a->shape.col;
	} else {
		shape_arr[0] = (float)a->shape.row;
		shape_arr[1] = (float)a->shape.col;
	}
	instance_free(a);
	coppia shape = {1, count};
	if (stack_push(my_stack, shape_arr, shape) != 0) {
		free(shape_arr);
		return -1;
	}
	return 0;
}


/* Pops a value tensor v (top) and a shape tensor s, pushes a new tensor of shape s
 * filled by cycling through the elements of v.
 * Validates dimensions >= 1 and product <= INT_MAX (entry point for user-supplied dims).
 * Input: my_stack — the stack (top: v, then s).
 * Output: 0 on success, -1 on error. */
int fill(stack *my_stack) {
	array_instance *v = stack_pop(my_stack);
	if (!v) return -1;
	array_instance *s = stack_pop(my_stack);
	if (!s) { instance_free(v); return -1; }

	if (s->shape.row != 1 || s->shape.col < 1 || s->shape.col > 2) {
		fprintf(stderr, "errore: fill richiede shape 1D di 1 o 2 elementi\n");
		instance_free(s); instance_free(v); return -1;
	}
	int row = (s->shape.col == 2) ? (int)s->data[0] : 1;
	int col = (s->shape.col == 2) ? (int)s->data[1] : (int)s->data[0];
	if (row < 1 || col < 1) {
		fprintf(stderr, "errore: fill richiede dimensioni >= 1\n");
		instance_free(s); instance_free(v); return -1;
	}
	instance_free(s);

	int64_t n64 = (int64_t)row * (int64_t)col;
	if (n64 > INT_MAX) {
		fprintf(stderr, "errore: tensore troppo grande\n");
		instance_free(v); return -1;
	}
	int n = (int)n64;
	int m = v->shape.row * v->shape.col;
	if (m <= 0) {
		fprintf(stderr, "fill: tensore valore non può essere vuoto\n");
		instance_free(v);
		return -1;
	}
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { perror("malloc"); instance_free(v); return -1; }

	for (int i = 0; i < n; i++)
		new_data[i] = v->data[i % m];

	instance_free(v);
	coppia shape = {row, col};
	if (stack_push(my_stack, new_data, shape) != 0) {
		free(new_data);
		return -1;
	}
	return 0;
}
