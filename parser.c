#include "parser.h"
#include "stack.h"

static dictionary table[] = {
	{"[", OP_NEW_ARRAY},
	{"p", OP_PRINT_N_POP},
	{"P", OP_PRINT_MATRIX}, // ormai lavevo scritta la mappo con P 
	{"d", OP_DUPLICATE},
	{"+", OP_ADD},
	{"-", OP_SUBTRACTION},
	{"*", OP_PRODUCT},
	{">", OP_GREATER},
	{"<", OP_LESSER},
	{"=", OP_CONFRONT},
	{"&", OP_AND},
	{"|", OP_OR}
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
 * Output: number of characters consumed (including ']'), or -1 on error.
 * Previously: two passes — first counted elements (malloc once), then re-read to fill.
 * Now: single pass with realloc at each element; slower on large arrays but simpler. */
int parse_array(char *s, int offset, stack *my_stack){
	int start = offset;
	int count = 0;
	float *new_array = NULL;
	int consumati = 0;

	while (s[offset] != '\0' && s[offset] != ']'){
		float val;
		if (sscanf(s + offset, " %f%n", &val, &consumati) == 1) { // magia nera di sscanf %f%n
			float *tmp = realloc(new_array, sizeof(float) * (size_t)(count + 1));
			if (tmp == NULL) {
				free(new_array);
				return -1;
			}
			new_array = tmp;
			new_array[count] = val;
			count++;
			offset += consumati;
		} else {
			offset++;
		}
	}

	if (s[offset] == '\0') {
		printf("errore di inserimento manca la chiusura ']' nella descrizione dell array");
		free(new_array);
		return -1;
	}

	coppia shape;
	shape.row = 1;
	shape.col = count;

	if (stack_push(my_stack, new_array, shape) != 0) {
		free(new_array);
		return -1;
	}
	return offset - start + 1;
}



/* Pops the top tensor and prints it in the format: Tensor(shape=[r c], data=[...]).
 * Input: my_stack — the stack.
 * Output: 0 on success. */
int pop_print(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	if (last == NULL) return -1;
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
	if (last == NULL) return -1;
	printf("on top of the stack was\n");
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
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
int duplicate (stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	if (last == NULL) return -1;
	if (stack_push_instance(my_stack, last) != 0) return -1;
	if (stack_push_instance(my_stack, last) != 0) return -1;
	return 0;
}

/* Pops the top two tensors, applies element-wise op ('a'=add, 's'=subtract, 'p'=product),
 * and pushes the result. Handles aliased operands (e.g. after duplicate) safely.
 * Input: my_stack — the stack, op — operation code.
 * Output: 0 on success. */
int algebra (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
			instance_free(a);
			if (b != a) instance_free(b);
			return -1;
	}
	float *new_data =  calloc((size_t)(a->shape.row * a->shape.col), sizeof(float));
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'a'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] + b->data[i * a->shape.col + j];
			}
		}
	} else if(op == 's'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] - b->data[i * a->shape.col + j];
			}
		}
	} else if(op == 'p'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] * b->data[i * a->shape.col + j];
			}
		}
	}
	coppia shape;
	shape.row = a->shape.row;
	shape.col = a->shape.col;
	
	instance_free(a);
	/* b può puntare allo stesso blocco di a se l'operando è stato duplicato con 'd'
	 * (duplicate non copia i dati, incrementa solo ref_count).
	 * In quel caso instance_free(a) ha già azzerato ref_count e liberato la memoria,
	 * quindi non dobbiamo liberare b una seconda volta. */
	if (b != a)
		instance_free(b);
	
	stack_push(my_stack,new_data,shape);
	return 0;
}

int disuguaglianze (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
			instance_free(a);
			if (b != a) instance_free(b);
			return -1;
	}
	float *new_data =  calloc((size_t)(a->shape.row * a->shape.col), sizeof(float));
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'M'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] > b->data[i * a->shape.col + j];
			}
		}
	} else if(op == 'm'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] < b->data[i * a->shape.col + j];
			}
		}
	} else if(op == 'u'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] == b->data[i * a->shape.col + j];
			}
		}
	}
	coppia shape;
	shape.row = a->shape.row;
	shape.col = a->shape.col;
	
	instance_free(a);
	/* b può puntare allo stesso blocco di a se l'operando è stato duplicato con 'd'
	 * (duplicate non copia i dati, incrementa solo ref_count).
	 * In quel caso instance_free(a) ha già azzerato ref_count e liberato la memoria,
	 * quindi non dobbiamo liberare b una seconda volta. */
	if (b != a)
		instance_free(b);
	
	stack_push(my_stack,new_data,shape);
	return 0;
}

int op_logiche_2_arg (stack *my_stack, char op) {
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
			instance_free(a);
			return -1;
	}
	for (int i = 0; i < a->shape.row; i++) {
		for (int j = 0; j < a->shape.col; j++) {
			if (a->data[i * a->shape.col + j] != 0.f && a->data[i * a->shape.col + j] != 1.f) {
				fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del primo tensore è %f\n", i, j, a->data[i * a->shape.col + j]);
				instance_free(a);
				if (b != a) instance_free(b);
				return -2;
			}
			if (b->data[i * a->shape.col + j] != 0.f && b->data[i * a->shape.col + j] != 1.f) {
				fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del secondo tensore è %f\n", i, j, b->data[i * a->shape.col + j]);
				instance_free(a);
				if (b != a) instance_free(b);
				return -2;
			}
		}
	}
	float *new_data =  calloc((size_t)(a->shape.row * a->shape.col), sizeof(float));
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'a'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] && b->data[i * a->shape.col + j];
			}
		}
	} else if(op == 'o'){
		for (int i = 0; i < a->shape.row; i++) {
			for (int j = 0; j < a->shape.col; j++) {
				new_data[i * a->shape.col + j] = a->data[i * a->shape.col + j] || b->data[i * a->shape.col + j];
			}
		}
	}
	coppia shape;
	shape.row = a->shape.row;
	shape.col = a->shape.col;
	
	instance_free(a);
	/* b può puntare allo stesso blocco di a se l'operando è stato duplicato con 'd'
	 * (duplicate non copia i dati, incrementa solo ref_count).
	 * In quel caso instance_free(a) ha già azzerato ref_count e liberato la memoria,
	 * quindi non dobbiamo liberare b una seconda volta. */
	if (b != a)
		instance_free(b);
	
	stack_push(my_stack,new_data,shape);
	return 0;
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
				break;}
			case OP_PRINT_N_POP: pop_print(my_stack); break;
			case OP_PRINT_MATRIX: pop_print_as_matrix(my_stack); break;
			case OP_DUPLICATE: duplicate(my_stack);  break;
			case OP_ADD: if (algebra(my_stack, 'a') != 0) return -1; break;
			case OP_SUBTRACTION: if (algebra(my_stack, 's') != 0) return -1; break;
			case OP_PRODUCT: if (algebra(my_stack, 'p') != 0) return -1; break;
			case OP_GREATER: if (disuguaglianze(my_stack, 'M') != 0) return -1; break;
			case OP_LESSER: if (disuguaglianze(my_stack, 'm') != 0) return -1; break;
			case OP_CONFRONT: if (disuguaglianze(my_stack, 'u') != 0) return -1; break;
			case OP_AND:{int err = op_logiche_2_arg(my_stack, 'a');
				if (err != 0) return err;
				break;
				}
			case OP_OR:{int err = op_logiche_2_arg(my_stack, 'o');
				if (err != 0) return err;
				break;
				}




			default:
				if (s[i] != ' ' && s[i] != '\n' && s[i] != '\t' && s[i] != '\r')
					printf("errore comando sconosciuto: '%c'\n", s[i]);
				break;		
		
		
		}
	}
	
	return 0 ;	
}
