#include "stack.h"
 /* Initializes a stack with an initial capacity of 1.
  * Returns NULL and prints an error if any allocation fails. */
stack* stack_init(void){
	stack *s = malloc(sizeof(stack));
	if (s == NULL){
		perror("malloc");
		return NULL;
	}
	s->top = 0;
	s->capacity = 1;
	s->stack = malloc((size_t)s->capacity*sizeof(array_instance*));
	if (s->stack == NULL){
		perror("malloc");
		free(s);
		return NULL;
	}
	return s;	
}

  /* Resizes the stack: doubles capacity when full, halves it when less than
   * half is used. Returns 0 on success, -1 on allocation failure. */
int stack_resize(stack *s){
	int32_t new_capacity;
	if (s->top == s->capacity)
		new_capacity = s->capacity * 2;
	else if (s->top < s->capacity / 2)
		new_capacity = s->capacity / 2;
	else
		return 0;
	array_instance **new_data = malloc((size_t)new_capacity * sizeof(array_instance*));
	if (new_data == NULL){
		perror("malloc");
		return -1;
	}
	memcpy(new_data, s->stack, (size_t)s->top*sizeof(array_instance*));
	free(s->stack);
	s->stack = new_data;
	s->capacity = new_capacity;
	return 0;
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
   * Returns 0 on success, -1 on allocation failure. */
int stack_push(stack *s, float *arr, coppia forma) {
    array_instance *instance = new_instance(arr, forma);
    if (instance == NULL)
        return -1;

    if (s->capacity == s->top)
        if (stack_resize(s) != 0) { free(instance); return -1; }

    s->stack[s->top++] = instance;
    return 0;
  }
  
/* Pushes an existing array_instance onto the stack, incrementing its ref_count.
 * Returns 0 on success, -1 on allocation failure. */
int stack_push_instance(stack *s, array_instance *inst) {
    if (s->capacity == s->top)
        if (stack_resize(s) != 0) return -1;
    inst->ref_count++;
    s->stack[s->top++] = inst;
    return 0;
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
    last->ref_count--;
    if (s->top < s->capacity / 2)
        stack_resize(s);
    return last;
}
	
	
	
/* Decrements ref_count and frees the instance if no references remain. */
void instance_free (array_instance *i){
	if (i == NULL) return;
	if (i->ref_count <= 0){
		free(i->data);
		free(i);
	}
}

/* Frees all instances on the stack, then frees the stack itself.
   * Respects ref_count: instances shared via dup/over are freed only when all references are gone. */
void stack_free(stack *s){
	while (s->top > 0){
		array_instance *inst = stack_pop(s);
		instance_free(inst);
	}
	free(s->stack);
	free(s);
}

