#ifndef STACK_DS
#define STACK_DS

#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#define MAX_DIM 2

typedef struct array_instance array_instance;

typedef struct {
	int32_t row;
	int32_t col;
} coppia;

struct array_instance {
	float *data;
	coppia shape;
	int32_t ref_count;
};

typedef struct {
	array_instance **stack;
	int32_t top;
	int32_t capacity;
} stack;

stack* stack_init (void); // implemented
void stack_resize(stack *s);
void stack_push(stack *s,float *arr, coppia shape);
void stack_push_instance(stack *s, array_instance *inst);
array_instance *stack_pop(stack *s);
void stack_free(stack *s);
array_instance *new_instance (float *data, coppia shape);
void instance_free (array_instance *i);
#endif 