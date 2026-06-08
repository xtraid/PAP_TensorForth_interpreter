#ifndef STACK_DS
#define STACK_DS

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_DIM 2

typedef enum {
  TF_OK = 0,
  TF_ERR_STACK  = -1, /* stack underflow or wrong item type */
  TF_ERR_SHAPE  = -2, /* incompatibol tensor shape */
  TF_ERR_TYPE   = -3, /* non-boolean value where hoolean required */
  TF_ERR_ARG    = -4, /* invalid arguments (dims, size, format) */
  TF_ERR_IO     = -5, /* file I/O error */
  TF_ERR_MEM    = -6, /* memory allocation failed */
  TF_ERR_SYNTAX = -7, /* script syntax error or invalid */
} TFError;

typedef struct array_instance array_instance;

typedef enum {ITEM_NONE, ITEM_TENSOR, ITEM_STRING} item_type;

typedef struct {
	item_type type;
	union {
		array_instance *tensor;
		char *filename;
	};	
}stack_item;

typedef struct {
	int32_t row;
	int32_t col;
  int32_t ndim;
} shape_t;

	
struct array_instance {
	float *data;
	shape_t shape;
	int32_t ref_count;
	int32_t on_disk;
	int32_t data_offset;
}; 

typedef struct {
	stack_item *stack;
	int32_t top;
	int32_t capacity;
} stack;

stack* stack_init (void); // implemented
int stack_resize(stack *s);
int stack_push(stack *s,float *arr, shape_t shape);
int stack_push_instance(stack *s, array_instance *inst);
array_instance *stack_pop(stack *s);
array_instance *stack_peek(stack *s);
void stack_free(stack *s);
array_instance *new_instance (float *data, shape_t shape);
void instance_free (array_instance *i);
int stack_push_string(stack *s, const char *filename);
stack_item stack_pop_item(stack *s);
int stack_push_item(stack *s, stack_item item);
void stack_free_item(stack_item item);


#endif 
