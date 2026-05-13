#ifndef STACK_DS
#define STACK_DS

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_DIM 2

typedef struct array_instance array_instance;

typedef enum {ITEM_TENSOR, ITEM_STRING }item_type;

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
} coppia;

	
struct array_instance {
	float *data;
	coppia shape;
	int32_t ref_count;
	int32_t on_disk;
	int32_t data_offset;
}; // 4 byte di padding 

typedef struct {
	stack_item *stack;
	int32_t top;
	int32_t capacity;
} stack;

stack* stack_init (void); // implemented
int stack_resize(stack *s);
int stack_push(stack *s,float *arr, coppia shape);
int stack_push_instance(stack *s, array_instance *inst);
array_instance *stack_pop(stack *s);
array_instance *stack_peek(stack *s);
void stack_free(stack *s);
array_instance *new_instance (float *data, coppia shape);
void instance_free (array_instance *i);
int stack_push_string(stack *s, const char *filename);
stack_item stack_pop_item(stack *s);


#endif 