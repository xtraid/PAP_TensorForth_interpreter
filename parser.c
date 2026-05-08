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
	{"|", OP_OR},
	{"!", OP_NOT},
	{"$", OP_MASK},
	{"@", OP_MAT_MUL},
	{"r", OP_RESHAPE},
	{"S", OP_SUM},
	{".", OP_DOT}

};



/* Looks up a token string in the command table.
 * Input: token — single-character null-terminated string.
 * Output: the corresponding OpCode, or OP_UNKNOWN if not found. */
// perf: const on token — communicates read-only intent, enables compiler alias analysis
OpCode lookup(const char *token){
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
// perf: const on s — communicates read-only intent, enables compiler alias analysis
int parse_array(const char *s, int offset, stack *my_stack){
	int start = offset;
	int count = 0;
	float *new_array = NULL;
	int consumati = 0;

	offset++;  // salta lo spazio obbligatorio dopo '['

	while (s[offset] != '\0' && s[offset] != ']') {
		float val;
		if (sscanf(s + offset, "%f%n", &val, &consumati) == 1) {
			float *tmp = realloc(new_array, sizeof(float) * (size_t)(count + 1));
			if (tmp == NULL) {
				free(new_array);
				return -1;
			}
			new_array = tmp;
			new_array[count] = val;
			count++;
			offset += consumati + 1;  // +1 = spazio dopo il float
		} else {
			printf("errore valore inserito non è un float: %c, continua la lettura ignorando il carattere\n", s[offset]);
			offset += 2;  // salta il char invalido + lo spazio dopo
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

/* Pops s (shape tensor: 1D, 1 or 2 elements) then a, reshapes a to the dimensions in s.
 * Memory layout is unchanged; only shape.row and shape.col are updated.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_reshape(stack *my_stack){
	array_instance *s = stack_pop(my_stack);
	if (s == NULL) return -1;
	if (s->shape.row != 1 || s->shape.col < 1 || s->shape.col > 2){
		fprintf(stderr, "errore: reshape richiede un tensore 1D di 1 o 2 elementi\n");
		instance_free(s);
		return -1;
	}
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) { instance_free(s); return -1; }

	int new_rows = (s->shape.col == 2) ? (int)s->data[0] : 1;
	int new_cols = (s->shape.col == 2) ? (int)s->data[1] : (int)s->data[0];

	if (new_rows * new_cols != a->shape.row * a->shape.col){
		fprintf(stderr, "errore: reshape incompatibile [%d %d] -> [%d %d]\n",
			a->shape.row, a->shape.col, new_rows, new_cols);
		instance_free(s);
		instance_free(a);
		return -1;
	}
	instance_free(s);
	a->shape.row = new_rows;
	a->shape.col = new_cols;
	if (stack_push_instance(my_stack, a) != 0){ instance_free(a); return -1; }
	instance_free(a);
	return 0;
}



/* Pops the top tensor and prints it in the format: Tensor(shape=[r c], data=[...]).
 * Input: my_stack — the stack.
 * Output: 0 on success. */
int pop_print(stack *my_stack){
	array_instance *last = stack_pop(my_stack);
	if (last == NULL) return -1;
	// perf: hoisted last->shape.row, last->shape.col, last->data — avoids repeated struct dereference in loop
	int rows      = last->shape.row;
	int cols      = last->shape.col;
	float *data   = last->data;
	int total = rows * cols;
	printf("Tensor(shape=[%d %d], data=[", rows, cols);
	for (int i = 0; i < total; i++){
		printf("%f", data[i]);
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
	// perf: hoisted last->shape.row, last->shape.col, last->data — avoids reload each iteration
	int rows    = last->shape.row;
	int cols    = last->shape.col;
	float *data = last->data;
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	for (int i = 0; i < rows; i++){
		printf("[ ");
		for (int j = 0; j < cols; j++){
			printf("%f ", data[i * cols + j]);
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
	if (stack_push_instance(my_stack, last) != 0) { instance_free(last); return -1; }
	if (stack_push_instance(my_stack, last) != 0) { instance_free(last); return -1; }
	instance_free(last);
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
	// perf: hoisted a->shape.row, a->shape.col, a->data, b->data — avoids reload each iteration
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	// perf: restrict on new_data — guarantees no aliasing with a_data/b_data, enables auto-vectorisation
	float * restrict new_data = calloc((size_t)(rows * cols), sizeof(float));
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'a'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] + b_data[i * cols + j];
			}
		}
	} else if(op == 's'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] - b_data[i * cols + j];
			}
		}
	} else if(op == 'p'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] * b_data[i * cols + j];
			}
		}
	}
	coppia shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	stack_push(my_stack,new_data,shape);
	return 0;
}

/* Pops the top two tensors and applies element-wise comparison op ('M'=greater, 'm'=lesser, 'u'=equal).
 * Result elements are 1.0 (true) or 0.0 (false). Tensors must have identical shape.
 * Input: my_stack — the stack, op — comparison code.
 * Output: 0 on success, -1 on error. */
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
	// perf: hoisted a->shape.row, a->shape.col, a->data, b->data — avoids reload each iteration
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	// perf: restrict on new_data — guarantees no aliasing with a_data/b_data, enables auto-vectorisation
	float * restrict new_data = calloc((size_t)(rows * cols), sizeof(float));
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'M'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] > b_data[i * cols + j];
			}
		}
	} else if(op == 'm'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] < b_data[i * cols + j];
			}
		}
	} else if(op == 'u'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] == b_data[i * cols + j];
			}
		}
	}
	coppia shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	stack_push(my_stack,new_data,shape);
	return 0;
}

/* Pops the top two tensors and applies element-wise logical op ('a'=AND, 'o'=OR).
 * Operands must be boolean (all elements 0.0 or 1.0) and have identical shape.
 * Input: my_stack — the stack, op — logical operation code.
 * Output: 0 on success, -1 on shape error, -2 on non-boolean element. */
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
	// perf: hoisted a->shape.row, a->shape.col, a->data, b->data — avoids reload each iteration
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (a_data[i * cols + j] != 0.f && a_data[i * cols + j] != 1.f) {
				fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del primo tensore è %f\n", i, j, a_data[i * cols + j]);
				instance_free(a);
				if (b != a) instance_free(b);
				return -2;
			}
			if (b_data[i * cols + j] != 0.f && b_data[i * cols + j] != 1.f) {
				fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del secondo tensore è %f\n", i, j, b_data[i * cols + j]);
				instance_free(a);
				if (b != a) instance_free(b);
				return -2;
			}
		}
	}
	// perf: restrict on new_data — guarantees no aliasing with a_data/b_data, enables auto-vectorisation
	float * restrict new_data = calloc((size_t)(rows * cols), sizeof(float));
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'a'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] && b_data[i * cols + j];
			}
		}
	} else if(op == 'o'){
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] || b_data[i * cols + j];
			}
		}
	}
	coppia shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	stack_push(my_stack,new_data,shape);
	return 0;
}

/* Pops the top tensor and applies element-wise logical NOT (0->1, 1->0).
 * Fails if any element is not 0.0 or 1.0.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_not (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	// perf: hoisted a->shape.row, a->shape.col, a->data — avoids reload each iteration
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	// perf: restrict on new_data — guarantees no aliasing with a_data, enables auto-vectorisation
	float * restrict new_data = calloc((size_t)(rows * cols), sizeof(float));
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			float val = a_data[i * cols + j];
			if (val == 0.f || val == 1.f){
				new_data[i * cols + j] = 1.f - val;
			} else {
				fprintf(stderr, "errore: not su array non booleano, elemento (%d, %d) = %f\n", i, j, val);
				free(new_data);
				instance_free(a);
				return -1;
			}
		}
	}
	coppia shape;
	shape.row = rows;
	shape.col = cols;
	instance_free(a);
	stack_push(my_stack, new_data, shape);
	return 0;
}

/* Pops m (mask), a, b from the stack; computes result[i] = m[i] ? a[i] : b[i].
 * m must be boolean (0.0/1.0); all three tensors must have identical shape.
 * Input: my_stack — the stack (top: m, then a, then b).
 * Output: 0 on success, -1 on error. */
int mask (stack *my_stack) {
	array_instance *m = stack_pop(my_stack);
	if (m == NULL) return -1;
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) return -1;

	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili a[%d %d] != b[%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		return -1;
	}
	if (a->shape.row != m->shape.row || a->shape.col != m->shape.col){
		fprintf(stderr, "errore: shape incompatibili a[%d %d] != m[%d %d]\n",
			a->shape.row, a->shape.col, m->shape.row, m->shape.col);
		return -1;
	}

	// non sono sicuro se "si assume che m sia booleano" intenda controllalo o non controllarlo io lo controllo nel caso a posteriori si puo commentare

	// perf: hoisted m->shape.row, m->shape.col — avoids reload each iteration
	int m_rows = m->shape.row;
	int m_cols = m->shape.col;
	for (int i = 0; i < m_rows; i++){
		for (int j = 0; j < m_cols; j++){
			if (m->data[i * m_cols + j] != 0.f && m->data[i * m_cols + j] != 1.f){
				fprintf(stderr, "errore: not su array non booleano, elemento (%d, %d) = %f\n", i, j, m->data[i * m_cols + j]);
				instance_free(m);
				instance_free(a);
				instance_free(b);
				return -1;

			}
		}
	}
	float *new_data = malloc(sizeof(float)*(size_t)(m_rows*m_cols));
	for (int i = 0; i < m_rows; i++){
		for (int j = 0; j < m_cols; j++){
			new_data[i * m_cols + j] = m->data[i * m_cols + j]*a->data[i * m_cols + j]+(1-m->data[i * m_cols + j])*b->data[i * m_cols + j];
		}
	}
	
	coppia shape;
	shape.row = m_rows;
	shape.col = m_cols;
	int err = stack_push(my_stack, new_data, shape);
	instance_free(m);
	instance_free(a);
	instance_free(b);
	return err;
}



/* Transposes a row-major matrix of shape [rows x cols] into new_data of shape [cols x rows].
 * new_data must be pre-allocated with rows*cols floats. */
static void transpose(float *data, float *new_data, int rows, int cols){
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			new_data[j * rows + i] = data[i * cols + j];
}

/* kernel e blocked_multiply ricevono B già trasposta (B_T shape [b_cols × a_cols]).
 * C[i][j] += A[i][k] * B_T[j][k]  =  C = A * B_originale
 * Entrambi gli accessi sono sequenziali in k -> cache friendly. */
static void kernel(float *A, float *B_T, float *C, int x, int dx, int y, int dy, int z, int dz, int a_rows, int a_cols, int b_cols){

	int mx = (x + dx > a_rows) ? a_rows : x + dx;
	int my = (y + dy > b_cols) ? b_cols : y + dy;
	int mz = (z + dz > a_cols) ? a_cols : z + dz;
	for (int i = x; i < mx; i++)
		for (int j = y; j < my; j++)
			for (int k = z; k < mz; k++)
				C[i * b_cols + j] += A[i * a_cols + k] * B_T[j * a_cols + k];
}

/* Multiplica A [a_rows × a_cols] per B_T [b_cols × a_cols] (già trasposta) e accumula in C [a_rows × b_cols].
 * Suddivide il lavoro in blocchi di dimensione s1×s2×s3 per migliorare la località di cache;
 * parallelizza i loop esterni con OpenMP. */
static void blocked_multiply(float *A, float *B_T, float *C, int a_rows, int a_cols, int b_cols){
	const int s1 = 16, s2 = 16, s3 = 16;
	#pragma omp parallel for collapse(2) schedule(dynamic)
	for (int i = 0; i < a_rows; i += s1)
		for (int j = 0; j < b_cols; j += s2)
			for (int k = 0; k < a_cols; k += s3)
				kernel(A, B_T, C, i, s1, j, s2, k, s3, a_rows, a_cols, b_cols);
}

/* Pops a (top) and b from the stack and pushes a@b (matrix product).
 * Requires a.cols == b.rows and rows > 1. Result shape is [a.rows x b.cols].
 * Transposes b internally for cache-friendly access, then uses blocked_multiply.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int mat_mat_mul(stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }

	int a_rows = a->shape.row;
	int a_cols = a->shape.col;
	int b_rows = b->shape.row;
	int b_cols = b->shape.col;

	if (a_cols != b_rows){
		fprintf(stderr, "errore: shape incompatibili per prodotto matriciale a[%d %d] b[%d %d]\n",
			a_rows, a_cols, b_rows, b_cols);
		instance_free(a);
		instance_free(b);
		return -1;
	}
	
	if (a_cols <= 1) {
		fprintf (stderr, " errore: le matrici devono essere bidimensionali");
		instance_free(a);
		instance_free(b);
		return -1;
	}

	float *b_T = malloc(sizeof(float) * (size_t)(b_rows * b_cols));
	if (b_T == NULL) { instance_free(a); instance_free(b); return -1; }
	transpose(b->data, b_T, b_rows, b_cols);

	float *new_data = calloc((size_t)(a_rows * b_cols), sizeof(float));
	if (new_data == NULL) { free(b_T); instance_free(a); instance_free(b); return -1; }

	blocked_multiply(a->data, b_T, new_data, a_rows, a_cols, b_cols);

	free(b_T);
	instance_free(a);
	instance_free(b);
	coppia shape = {a_rows, b_cols};
	return stack_push(my_stack, new_data, shape);
}


/* Pops the top tensor and pushes a scalar [1×1] with the sum of all its elements.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int sum_arr (stack *my_stack){
	array_instance *arr = stack_pop(my_stack);
	if (arr == NULL) return -1;
	int n = arr->shape.row * arr->shape.col;
	float *sum = malloc (sizeof(float));
	sum[0]= 0;
	
	for (int i = 0; i < n; i++)
		sum[0] += arr->data[i];
	
	instance_free(arr);
	
	coppia shape;
	shape.row = 1;
	shape.col = 1;
	stack_push(my_stack, sum, shape);
	return 0;
}


/* Pops two 1D vectors (shape [1×n]) and pushes a scalar [1×1] with their dot product.
 * Both operands must be row vectors (shape.row == 1) of equal length.
 * Input: my_stack — the stack (top: a, then b).
 * Output: 0 on success, -1 on error. */
int dot_product (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) {
		instance_free (a);
		return -1;
	}
	if (a->shape.row > 1) {
		fprintf (stderr, " errore: il primo argomento non è un vettore");
		return -1;	
	}
	if (b->shape.row > 1) {
		fprintf (stderr, " errore: il secondo argomento non è un vettore");
		return -1;	
	}
	int n = a->shape.col;
	float *sum = malloc (sizeof(float));
	sum[0]= 0;
	
	for (int i = 0; i < n; i++)
		sum[0] += a->data[i] * b->data[i];
	
	instance_free(a);
	instance_free(b);
	
	coppia shape;
	shape.row = 1;
	shape.col = 1;
	stack_push(my_stack, sum, shape);
	return 0;
}



/* Main interpreter loop. Scans s token by token and dispatches each command.
 * Input: s — null-terminated script string, my_stack — the execution stack.
 * Output: 0 on success, -1 on error. */
// perf: const on s — communicates read-only intent, enables compiler alias analysis
int parser(const char *s, stack *my_stack){
	long size = (long)strlen(s);
	for (int i = 0; i < size; i++){
		char token[2] = {s[i], '\0'};
 		switch(lookup(token)){

			case OP_NEW_ARRAY:{
				int result = parse_array(s, i+1, my_stack);
				if (result == -1) return -1;
					i += result;
				break;}
			case OP_PRINT_N_POP: if (pop_print(my_stack) != 0) return -1; break;
			case OP_PRINT_MATRIX: if (pop_print_as_matrix(my_stack) != 0) return -1; break;
			case OP_DUPLICATE: if (duplicate(my_stack) != 0) return -1; break;
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
			case OP_NOT:{int err = op_not(my_stack);
				if (err != 0) return err;
				break;
				}
			case OP_MASK:{int err = mask(my_stack);
				if (err != 0) return err;
				break;
				}
			case OP_MAT_MUL:{int err = mat_mat_mul(my_stack);
				if (err != 0) return err;
				break;
				}
			case OP_RESHAPE:{int err = op_reshape(my_stack);
				if (err != 0) return err;
				break;
				}
			case OP_SUM: if (sum_arr(my_stack) != 0) return -1; break;
			case OP_DOT: if (dot_product(my_stack) != 0) return -1; break;

			default:
				if (s[i] != ' ' && s[i] != '\n' && s[i] != '\t' && s[i] != '\r')
					printf("errore comando sconosciuto: '%c'\n", s[i]);
				break;
			


		}
	}

	return 0 ;
}
