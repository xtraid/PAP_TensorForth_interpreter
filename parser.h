#ifndef PARSER_
#define PARSER_

#include <stdint.h>
#include <sys/types.h>
#include "stack.h"

typedef struct {
	int32_t shape[MAX_DIM];
	int32_t ndim;
	off_t   data_offset;
} on_disk_tensor;

typedef enum {
	//stack functions
	OP_NEW_ARRAY,
	OP_PRINT_N_POP,
	OP_PRINT_MATRIX,
	OP_DUPLICATE,
	OP_ADD,
	OP_SUBTRACTION,
	OP_PRODUCT,
	OP_GREATER,
	OP_LESSER,
	OP_CONFRONT,
	OP_AND,
	OP_OR,
	OP_NOT,
	OP_MASK,
	OP_MAT_MUL,
	OP_RESHAPE,
	OP_SUM,
	OP_DOT,
	OP_READ_NAME,
	OP_LOAD_TENSOR,
	OP_SAVE_TENSOR,
	OP_LOAD_MMAP,
	OP_SAVE_MMAP,

	OP_UNKNOWN
}OpCode;

OpCode lookup(const char *token);

int parser(const char *s, stack *my_stack);

long parse_array(const char *s, long offset, stack *my_stack);

long parse_string(const char *s, long offset, stack *my_stack);

int read_image(stack *my_stack, char op);

int load_mmap(stack *my_stack);

int save_mmap(stack *my_stack);

int pop_print(stack *my_stack);

int pop_print_as_matrix(stack *my_stack);

int duplicate (stack *my_stack);

int algebra (stack *my_stack, char op);

int op_logiche_2_arg (stack *my_stack, char op);

int op_not (stack *my_stack);

int mask (stack *my_stack);

int mat_mat_mul(stack *my_stack);

int op_reshape(stack *my_stack);

int disuguaglianze(stack *my_stack, char op);

int sum_arr(stack *my_stack);

int dot_product(stack *my_stack);

#endif
