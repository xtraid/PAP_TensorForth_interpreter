#include "stack.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static float *padding(float *data, int row, int col, int pad_dim) {
    int row_padded = row + (2 * pad_dim);
    int col_padded = col + (2 * pad_dim);

    float *new_data = calloc((size_t)(row_padded * col_padded), sizeof(float));
    if (new_data == NULL) {
        return NULL;
    }

    for (int r = 0; r < row; r++) {
        memcpy(&new_data[(r + pad_dim) * col_padded + pad_dim], &data[r * col], sizeof(float) * (size_t)col);
    }

    return new_data;
}

static float c_dot(float *window_start, float *k, int stride, int shape_k) {
    float sum = 0.0f;
    for (int r = 0; r < shape_k; r++) {
        for (int c = 0; c < shape_k; c++) {
            sum += window_start[r * stride + c] * k[r * shape_k + c];
        }
    }
    return sum;
}

int convoluzione(stack *my_stack) {
    array_instance *k = stack_pop(my_stack);
    if (!k)
        return TF_ERR_STACK;

    if (k->shape.row < 2 || k->shape.col < 2) {
        fprintf(stderr, "error: 'c': kernel must be a 2D matrix\n");
        instance_free(k);
        return TF_ERR_ARG;
    }

    if (k->shape.col != k->shape.row || k->shape.col % 2 == 0) {
        fprintf(stderr, "error: 'c': kernel must be a square matrix with odd dimensions\n");
        instance_free(k);
        return TF_ERR_ARG;
    }

    array_instance *a = stack_pop(my_stack);
    if (!a) {
        instance_free(k);
        return TF_ERR_STACK;
    }

    int pad_dim = (k->shape.row - 1) / 2;
    float *expanded_data = padding(a->data, a->shape.row, a->shape.col, pad_dim);
    if (!expanded_data) {
        instance_free(k);
        instance_free(a);
        return TF_ERR_MEM;
    }

    int col_padded = a->shape.col + 2 * pad_dim;
    int shape_k = k->shape.row;
    int a_rows = a->shape.row;
    int a_cols = a->shape.col;

    float *out = malloc(sizeof(float) * (size_t)(a_rows * a_cols));
    if (!out) { free(expanded_data); instance_free(k); instance_free(a); return TF_ERR_MEM; }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            float *window = &expanded_data[i * col_padded + j];
            out[i * a_cols + j] = c_dot(window, k->data, col_padded, shape_k);
        }
    }

    free(expanded_data);
    instance_free(k);
    instance_free(a);

    shape_t shape = {a_rows, a_cols, 2};
    if (stack_push(my_stack, out, shape) != 0) { free(out); return TF_ERR_MEM; }
    return TF_OK;
}
