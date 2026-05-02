#include "parser.h"
#include "stack.h"

static dictionary table[] = {
	{"[", OP_NEW_ARRAY},
	{"p", OP_PRINT_N_POP},
	{"P", OP_PRINT_MATRIX}, // ormai lavevo scritta la mappo con P 
	{"d", OP_DUPLICATE}
};



/* Looks up a token string in the command table.
 * Input: token — single-character null-terminated string.
 * Output: the corresponding OpCode, or OP_UNKNOWN if not found. */
OpCode lookup(char *token){
	int n = sizeof(table)/sizeof(table[0]);
	for (int i = 0; i < n; i++){
		if(strcmp(table[i].nome, token)== 0)
			return table[i].op;
	}
	return OP_UNKNOWN;
}

/* Parses a 1D float array literal starting after '[' at s[offset].
 * Reads floats until ']' is found, allocates the array and pushes it on the stack.
 * Input: s — full script string, offset — index of first char after '[', my_stack — destination stack.
 * Output: number of characters consumed (including ']'), or -1 on error. */
int parse_array(char *s, int offset, stack *my_stack){
	int start = offset;
	int count = 0;
	int tmp =offset;
	while (s[tmp] != '\0' && s[tmp] != ']'){
		float val;
		int consumati = 0;
		if (sscanf(s + tmp, " %f%n", &val, &consumati) == 1) { // magia nera di sscanf %f%n
			count++;
			tmp += consumati;
		}
		else {
			tmp++;
		}		
	}
	if (s[tmp] == '\0') {
		printf("errore di inserimento manca la chiusura ']' nella descrizione dell array");
		return -1;		
	}
	int i = tmp; // posizione di ']'
	float *new_array = malloc(sizeof(float)*(size_t)count);
	int j = 0;
	int consumati = 0;
	while(j < count){
		float val;
		if (sscanf(s+offset, " %f%n", &val, &consumati) == 1) {
			new_array[j] = val;
			j++;
			offset += consumati;
		}
		else {
			printf("errore di inserimento: %c, %c", s[offset], s[offset+1]);
			free (new_array);
			return -1;
		}
		
	}
	coppia shape;
	shape.row = 1;
	shape.col = j;
	
	stack_push(my_stack, new_array, shape);
	return i - start +1;
}



/* Pops the top tensor and prints it in the format: Tensor(shape=[r c], data=[...]).
 * Input: my_stack — the stack.
 * Output: 0 on success. */
int pop_print(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	int total = last->shape.row * last->shape.col;
	printf("Tensor(shape=[%d %d], data=[", last->shape.row, last->shape.col);
	for (int i = 0; i < total; i++){
		printf("%f", last->data[i]);
		if (i < total - 1) printf(" ");
	}
	printf("])\n");
	instance_free(last);
	return 0;
}

/* Pops the top tensor and prints it row by row for visual debugging.
 * Input: my_stack — the stack.
 * Output: 0 on success. */
int pop_print_as_matrix(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	printf("on top of the stack was\n");
	for (int i = 0; i < last->shape.row; i++){
		printf("[ ");
		for (int j = 0; j < last->shape.col; j++){
			printf("%f ", last->data[i * last->shape.col + j]);
		}
		printf("]\n");
	}
	instance_free(last);
	return 0;
}



/* Duplicates the top element of the stack by incrementing its ref_count.
 * Does not allocate a new tensor — both stack entries point to the same instance.
 * Input: my_stack — the stack. */
void duplicate (stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	stack_push_instance(my_stack, last);
	stack_push_instance(my_stack, last);
}



/* Main interpreter loop. Scans s token by token and dispatches each command.
 * Input: s — null-terminated script string, my_stack — the execution stack.
 * Output: 0 on success, -1 on error. */
int parser(char *s,stack *my_stack){
	long size = (long)strlen(s);
	for (int i = 0; i < size; i++){
		char token[2] = {s[i], '\0'};
 		switch(lookup(token)){
			case OP_NEW_ARRAY:{ 
			int result = parse_array(s, i+1, my_stack);
			if (result == -1) return -1;
				i += result;
				if (sscanf(s + i, "] ") != 0)
					printf("errore manca la chiusura della parentesi");
			break;}
			
			case OP_PRINT_N_POP: pop_print(my_stack); break;
			case OP_PRINT_MATRIX: pop_print_as_matrix(my_stack); break;
			case OP_DUPLICATE: duplicate(my_stack);  break;

			default:
				if (s[i] != ' ' && s[i] != '\n' && s[i] != '\t' && s[i] != '\r')
					printf("errore comando sconosciuto: '%c'\n", s[i]);
				break;		
		
		
		}
	}
	
	return 0 ;
	
	
}
