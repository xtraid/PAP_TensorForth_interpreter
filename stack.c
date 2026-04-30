#include "stack.h"
 /* Initializes a stack with an initial capacity of 1.
  * Returns NULL and prints an error if any allocation fails. */
stack* stack_init(){
	stack *s = malloc(sizeof(stack));
	if (s == NULL){
		perror("malloc");
		return NULL;
	}
	s->top = 0;
	s->capacity = 1;
	s->stack = malloc(s->capacity*sizeof(array_instance*));
	if (s->stack == NULL){
		perror("malloc");
		free(s);
		return NULL;
	}
	return s;	
}

  /* Resizes the stack: doubles capacity when full, halves it when less than
   * half is used. Returns silently on allocation failure. */
void stack_resize(stack *s){
	int32_t new_capacity;
	if (s->top == s->capacity)
		new_capacity = s->capacity * 2;
	else if (s->top < s->capacity / 2)
		new_capacity = s->capacity / 2;
	else 
		return;
	array_instance **new_data = malloc(new_capacity * sizeof(array_instance*));
	if (new_data == NULL){
		perror("malloc");
		return;
	}
	memcpy(new_data, s->stack, s->top*sizeof(array_instance*));
	free(s->stack);
	s->stack = new_data;
	s->capacity = new_capacity;
}

/* Allocates and initializes a new array_instance with ref_count 1.
  * Returns NULL on allocation failure. */
array_instance *new_instance(float *data, coppia shape) {
    array_instance *instance = malloc(sizeof(array_instance));
    if (instance == NULL) {
        perror("malloc");
        return NULL;
    }
    instance->data = data;
    instance->shape = shape;
    instance->ref_count = 1;
    return instance;
  }

  /* Pushes a new tensor onto the stack. Resizes if at capacity.
   * Returns silently on allocation failure. */
void stack_push(stack *s, float *arr, coppia forma) {
    array_instance *instance = new_instance(arr, forma);
    if (instance == NULL)
        return;

    if (s->capacity == s->top)
       stack_resize(s);
	
    s->stack[s->top++] = instance;
  }
  
/* Pops and returns the top element of the stack.
   * Shrinks the stack if less than half capacity is used.
   * Caller is responsible for calling release_instance when done. */
array_instance *stack_pop(stack *s){
	if (s->top == 0){
        fprintf(stderr, "stack underflow\n");
		return NULL;
		}
    array_instance *last = s->stack[--s->top];
    if (s->top < s->capacity / 2)
        stack_resize(s);
    return last;
}
	
	
	
/* Decrements ref_count and frees the instance if no references remain. */
void instance_free (array_instance *i){
	if (i == NULL) return;
	if (--i->ref_count ==  0){
		free(i->data);
		free(i);		
	}	
}

/* Frees all instances on the stack, then frees the stack itself.
   * Respects ref_count: instances shared via dup/over are freed only when all references are gone. */
void stack_free(stack *s){
	for (int i = 0; i < s->top; i++){
		instance_free(s->stack[i]);
	}
	free(s->stack);
	free(s);
	
	
}

