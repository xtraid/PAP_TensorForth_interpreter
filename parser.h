#ifndef PARSER_
#define PARSER_

#include "stack.h"
#include <stdint.h>
#include <sys/types.h>



typedef struct {
	int32_t shape[MAX_DIM];
	int32_t ndim;
	off_t data_offset;
} disk_header;

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
	OP_LOAD_PGM,
	OP_SAVE_PGM,
	OP_RANDOM,
	OP_RELU,
	OP_MIN,
	OP_MAX,
	OP_SWITCH,
	OP_OVER,
	OP_DROP,
	OP_RAVEL,
	OP_SHAPE,
	OP_FILL,
	OP_SAVE_DISK,
	OP_LOAD_DISK,
	OP_CONV,

	OP_UNKNOWN
}OpCode;

OpCode lookup(const char *token);

int parser(const char *s, stack *my_stack);

int random_array(stack *my_stack);

int relu(stack *my_stack);

int extrema(stack *my_stack, char op);

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

int op_switch(stack *my_stack);

int over(stack *my_stack);

int drop(stack *my_stack);

int ravel(stack *my_stack);

int op_shape(stack *my_stack);

int fill(stack *my_stack);

int on_disk_save(stack *my_stack);
int on_disk_read(stack *my_stack);
int op_conv(stack *my_stack);

#endif
