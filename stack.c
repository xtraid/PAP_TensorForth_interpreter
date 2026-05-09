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
	s->stack = malloc((size_t)s->capacity*sizeof(stack_item));
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
	else if (s->top < s->capacity / 4)
		new_capacity = s->capacity / 2;
	else
		return 0;
	stack_item *new_data = malloc((size_t)new_capacity * sizeof(stack_item));
	if (new_data == NULL){
		perror("malloc");
		return -1;
	}
	memcpy(new_data, s->stack, (size_t)s->top * sizeof(stack_item));
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
    instance->on_disk = 0;
    instance->data_offset = 0;
    return instance;
  }

  /* Pushes a new tensor onto the stack. Resizes if at capacity.
   * Returns 0 on success, -1 on allocation failure. */
int stack_push(stack *s, float *arr, coppia forma) {
    array_instance *instance = new_instance(arr, forma);
    if (instance == NULL)
        return -1;

    if (s->capacity == s->top){
        if (stack_resize(s) != 0) { free(instance); return -1; }
	}

	stack_item item;
	item.type = ITEM_TENSOR;
	item.tensor = instance;
    s->stack[s->top++] = item;
    return 0;
  }
  
/* Pushes an existing array_instance onto the stack, incrementing its ref_count.
 * Returns 0 on success, -1 on allocation failure. */
int stack_push_instance(stack *s, array_instance *inst) {
    if (s->capacity == s->top)
        if (stack_resize(s) != 0) return -1;
    inst->ref_count++;
	stack_item item;
	item.type = ITEM_TENSOR;
	item.tensor = inst;
    s->stack[s->top++] = item;
    return 0;
}

/* Pops and returns the top element of the stack.
   * Shrinks the stack if less than a quarter of capacity is used.
   * Caller is responsible for calling instance_free when done. */
array_instance *stack_pop(stack *s){
	if (s->top == 0){
        fprintf(stderr, "stack underflow\n");
		return NULL;
		}
    stack_item item = s->stack[--s->top];
    if (item.type == ITEM_STRING) {
        fprintf(stderr, "errore: atteso tensore, trovato stringa \"%s\"\n", item.filename);
        free(item.filename);
        if (s->top < s->capacity / 4) stack_resize(s);
        return NULL;
    }
    if (s->top < s->capacity / 4)
        stack_resize(s);
    return item.tensor;
}
	
	
	
/* Decrements ref_count and frees the instance if no references remain. */
void instance_free (array_instance *i){
	if (i == NULL) return;
	i->ref_count--;
	if (i->ref_count <= 0){
		free(i->data);
		free(i);
	}
}

/* Frees all instances on the stack, then frees the stack itself.
   * Respects ref_count: instances shared via dup/over are freed only when all references are gone. */
void stack_free(stack *s){
	while (s->top > 0){
		stack_item item = stack_pop_item(s);
        if (item.type == ITEM_TENSOR)
            instance_free(item.tensor);
        else
            free(item.filename);
	}
	free(s->stack);
	free(s);
}
/* Pushes a filename string onto the stack. Makes a copy of the string.
   * Returns 0 on success, -1 on allocation failure. */
int stack_push_string(stack *s, const char *filename){
	if (s->capacity == s->top)
		if(stack_resize(s) != 0) return -1;
	stack_item item;
	item.type = ITEM_STRING;
	item.filename = malloc(strlen(filename) + 1);
	if (item.filename == NULL) { perror("malloc"); return -1; }
	strcpy(item.filename, filename);
	s->stack[s->top++] = item;
	return 0;	
}


/* Pops and returns the top stack_item (tensor or string).
   * Caller must free item.filename if type == ITEM_STRING. */
  stack_item stack_pop_item(stack *s) {
	stack_item empty = {0};
	if (s->top == 0) { 
		fprintf(stderr, "stack underflow\n"); return empty;
	}
	stack_item item = s->stack[--s->top];
	if (s->top < s->capacity / 4)
        stack_resize(s);
	return item;
  }