#ifndef PARSER_
#define PARSER_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stack.h"



typedef enum {
	//stack functions
	OP_NEW_ARRAY,
	OP_PRINT_N_POP,
	OP_PRINT_MATRIX,
	OP_DUPLICATE,
	OP_ADD,
	OP_SUBTRACTION,
	OP_PRODUCT,

	OP_UNKNOWN
}OpCode;

typedef struct{
	char *nome;
	OpCode op;	
}dictionary;
 
OpCode lookup(char *token);

int parser(char *s, stack *my_stack);

int parse_array(char *arr, int offset, stack *my_stack);

int pop_print(stack *my_stack);

int pop_print_as_matrix(stack *my_stack);

int duplicate (stack *my_stack);

int algebra (stack *my_stack, char op);


#endif
