#ifndef _STACK_DS
#define _STACK_DS

#include <stdint.h>

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
	array_instance **data;
	int32_t top;
	int32_t capacity;
} stack;

stack* stack_init ();
int stack_push(stack *s,float *data);
array_instance * stack_pop(stack *s);
void stack_free(stack *s);
array_instance * new_instance (float *data, coppia shape);
void instance_free (array_instance *i);
#endif 