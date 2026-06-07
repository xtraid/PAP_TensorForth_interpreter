#include "parser.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#define BLOCK_SIZE 64

typedef struct { char *nome; OpCode op; } dictionary;

static dictionary table[] = {
	{"[", OP_NEW_ARRAY},
	{"p", OP_PRINT_N_POP},
	{"P", OP_PRINT_MATRIX},
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
	{".", OP_DOT},
	{"\"", OP_READ_NAME},
	{"(", OP_LOAD_PGM},
	{")", OP_SAVE_PGM},
	{"?", OP_RANDOM},
	{"R", OP_RELU},
	{"m", OP_MIN},
	{"M", OP_MAX},
	{"s", OP_SWITCH},
	{"o", OP_OVER},
	{"D", OP_DROP},
	{"_", OP_RAVEL},
	{"#", OP_SHAPE},
	{"f", OP_FILL},
	{"}", OP_SAVE_DISK},
	{"{", OP_LOAD_DISK},
	{"c", OP_CONV}

};



/* Looks up a token string in the command table.
 * Input: token — single-character null-terminated string.
 * Output: the corresponding OpCode, or OP_UNKNOWN if not found. */
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
 * Output: number of characters consumed (including ']'), or -1 on error. */
static long parse_array(const char *s, long offset, stack *my_stack){
	long start = offset;
	int count = 0;
	int consumati = 0;
	long tmp = offset;

	/* prima passata: valida la sintassi e conta gli elementi */
	if (s[tmp] != ' ') {
		fprintf(stderr, "errore: manca lo spazio obbligatorio dopo '['\n");
		return -1;
	}
	tmp++;

	while (s[tmp] != '\0' && s[tmp] != ']') {
		if (s[tmp] == ' ') {
			fprintf(stderr, "errore: spazio doppio alla posizione %ld\n", tmp - start);
			return -1;
		}
		float val;
		if (sscanf(s + tmp, "%f%n", &val, &consumati) == 1) {
			if (s[tmp + consumati] != ' ') {
				if (s[tmp + consumati] == ']')
					fprintf(stderr, "errore: manca lo spazio obbligatorio prima di ']'\n");
				else
					fprintf(stderr, "errore: manca lo spazio dopo il valore %g\n", val);
				return -1;
			}
			count++;
			tmp += consumati + 1;
		} else {
			if (isprint((unsigned char)s[tmp]))
				fprintf(stderr, "errore: token non valido '%c' alla posizione %ld\n", s[tmp], tmp - start);
			else
				fprintf(stderr, "errore: token non valido '\\x%02x' alla posizione %ld\n", (unsigned char)s[tmp], tmp - start);
			return -1;
		}
	}

	if (s[tmp] == '\0') {
		fprintf(stderr, "errore: manca la chiusura ']'\n");
		return -1;
	}

	long end = tmp;  /* posizione di ']' */

	if (count == 0) {
		fprintf(stderr, "errore: array vuoto '[ ]' non supportato\n");
		return -1;
	}

	/* seconda passata: alloca esattamente `count` elementi e riempie.
	 * Il doppio scan è deliberato: la prima passata conta gli elementi per evitare
	 * realloc su array di dimensione arbitraria. */
	float *new_array = malloc(sizeof(float) * (size_t)count);
	if (new_array == NULL) return -1;

	offset++;  /* salta lo spazio iniziale dopo '[' */
	for (int j = 0; j < count; j++) {
		float val;
		if (sscanf(s + offset, "%f%n", &val, &consumati) == 1) {
			new_array[j] = val;
			offset += consumati + 1;
		} else {
			free(new_array);
			return -1;
		}
	}

	coppia shape;
	shape.row = 1;
	shape.col = count;

	if (stack_push(my_stack, new_array, shape) != 0) {
		free(new_array);
		return -1;
	}
	return end - start + 1;
}

/* Pops s (shape tensor: 1D, 1 or 2 elements) then a, reshapes a to the dimensions in s.
 * Memory layout is unchanged; only shape.row and shape.col are updated.
 * Validates dimensions >= 1 and product <= INT_MAX (entry point for user-supplied dims).
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

	if (new_rows < 1 || new_cols < 1) {
		fprintf(stderr, "errore: reshape richiede dimensioni >= 1\n");
		instance_free(s); instance_free(a); return -1;
	}
	int64_t n64 = (int64_t)new_rows * (int64_t)new_cols;
	if (n64 > INT_MAX) {
		fprintf(stderr, "errore: tensore troppo grande\n");
		instance_free(s); instance_free(a); return -1;
	}
	if (new_rows * new_cols != a->shape.row * a->shape.col){
		fprintf(stderr, "errore: reshape incompatibile [%d %d] -> [%d %d]\n",
			a->shape.row, a->shape.col, new_rows, new_cols);
		instance_free(s);
		instance_free(a);
		return -1;
	}
	instance_free(s);
	int n = new_rows * new_cols;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); return -1; }
	memcpy(new_data, a->data, sizeof(float) * (size_t)n);
	instance_free(a);
	coppia shape = {new_rows, new_cols};
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
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
	array_instance *top = stack_peek(my_stack);
	if (top == NULL) return -1;
	return stack_push_instance(my_stack, top);
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
			/* instance_free usa ref_count: chiamarlo due volte è sicuro anche se a==b (2→1→0) */
			instance_free(b);
			return -1;
	}
	// perf: hoisted a->shape.row, a->shape.col, a->data, b->data — avoids reload each iteration
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	// perf: restrict on new_data — guarantees no aliasing with a_data/b_data, enables auto-vectorisation
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); instance_free(b); return -1; }
	/* row-major: elemento (i,j) -> data[i * shape.col + j] */
	if (op == 'a'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] + b_data[i * cols + j];
			}
		}
	} else if(op == 's'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] - b_data[i * cols + j];
			}
		}
	} else if(op == 'p'){
		#pragma omp parallel for collapse(2) schedule(static)
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

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
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
			instance_free(b);
			return -1;
	}
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); instance_free(b); return -1; }
	if (op == 'M'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] > b_data[i * cols + j];
			}
		}
	} else if(op == 'm'){
		#pragma omp parallel for collapse(2) schedule(static)
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				new_data[i * cols + j] = a_data[i * cols + j] < b_data[i * cols + j];
			}
		}
	} else if(op == 'u'){
		#pragma omp parallel for collapse(2) schedule(static)
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

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
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
			instance_free(b);
			return -1;
	}
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict b_data = b->data;
	int n = rows * cols;
	float * restrict new_data = malloc(sizeof(float) * (size_t)n);
	if (new_data == NULL) { instance_free(a); instance_free(b); return -1; }
	for (int k = 0; k < n; k++) {
		if (a_data[k] != 0.f && a_data[k] != 1.f) {
			fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del primo tensore è %f\n", k / cols, k % cols, a_data[k]);
			free(new_data); instance_free(a); instance_free(b);
			return -2;
		}
		if (b_data[k] != 0.f && b_data[k] != 1.f) {
			fprintf(stderr, "operazione logica non definita: elemento [%d, %d] del secondo tensore è %f\n", k / cols, k % cols, b_data[k]);
			free(new_data); instance_free(a); instance_free(b);
			return -2;
		}
	}
	#pragma omp parallel for schedule(static)
	for (int k = 0; k < n; k++)
		new_data[k] = (op == 'a') ? (float)(a_data[k] && b_data[k]) : (float)(a_data[k] || b_data[k]);
	coppia shape;
	shape.row = rows;
	shape.col = cols;

	instance_free(a);
	instance_free(b);

	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}

/* Pops the top tensor and applies element-wise logical NOT (0->1, 1->0).
 * Fails if any element is not 0.0 or 1.0.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_not (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	int rows = a->shape.row;
	int cols = a->shape.col;
	float * restrict a_data = a->data;
	float * restrict new_data = malloc(sizeof(float) * (size_t)(rows * cols));
	if (new_data == NULL) { instance_free(a); return -1; }
	int n_not = rows * cols;
	for (int i = 0; i < n_not; i++) {
		if (a_data[i] != 0.f && a_data[i] != 1.f) {
			fprintf(stderr, "errore: not su array non booleano, elemento (%d, %d) = %f\n", i / cols, i % cols, a_data[i]);
			free(new_data); instance_free(a); return -1;
		}
	}
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_not; i++)
		new_data[i] = 1.f - a_data[i];
	coppia shape;
	shape.row = rows;
	shape.col = cols;
	instance_free(a);
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
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
	if (a == NULL) { 
		instance_free(m); return -1;
		}
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { 
		instance_free(m);
		instance_free(a); 
		return -1; }

	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili a[%d %d] != b[%d %d]\n",
			a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(m); instance_free(a); instance_free(b);
		return -1;
	}
	if (a->shape.row != m->shape.row || a->shape.col != m->shape.col){
		fprintf(stderr, "errore: shape incompatibili a[%d %d] != m[%d %d]\n",
			a->shape.row, a->shape.col, m->shape.row, m->shape.col);
		instance_free(m); instance_free(a); instance_free(b);
		return -1;
	}

	int m_rows = m->shape.row;
	int m_cols = m->shape.col;
	for (int i = 0; i < m_rows; i++){
		for (int j = 0; j < m_cols; j++){
			if (m->data[i * m_cols + j] != 0.f && m->data[i * m_cols + j] != 1.f){
				fprintf(stderr, "errore: mask su maschera non booleana, elemento (%d, %d) = %f\n", i, j, m->data[i * m_cols + j]);
				instance_free(m);
				instance_free(a);
				instance_free(b);
				return -1;

			}
		}
	}
	float *new_data = malloc(sizeof(float)*(size_t)(m_rows*m_cols));
	if (new_data == NULL) { 
		instance_free(m);
		instance_free(a);
		instance_free(b);
		return -1; 
		}
	/* Precedente: select aritmetico m*a + (1-m)*b.
	 * Rimosso perché inf*0 = NaN contamina il risultato anche nel ramo non scelto.
	 * Il ternario genera cmov con -O3, è branchless e corretto su inf/NaN. */
	int nm = m_rows * m_cols;
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < nm; i++)
		new_data[i] = (m->data[i] != 0.0f) ? a->data[i] : b->data[i];
	
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
static void kernel(float *A, float *B_T, float *C,
                   int x, int s1, int y, int s2, int k, int s3,
                   int a_rows, int a_cols, int b_cols){
    int mx = x + s1 < a_rows ? x + s1 : a_rows;
    int my = y + s2 < b_cols ? y + s2 : b_cols;
    int mk = k + s3 < a_cols ? k + s3 : a_cols;
    for (int i = x; i < mx; i++)
        for (int j = y; j < my; j++) {
            float sum = C[i * b_cols + j];
            for (int kk = k; kk < mk; kk++)
                sum += A[i * a_cols + kk] * B_T[j * a_cols + kk];
            C[i * b_cols + j] = sum;
        }
}

/* Multiplica A [a_rows × a_cols] per B_T [b_cols × a_cols] (già trasposta) e accumula in C [a_rows × b_cols].
 * Suddivide il lavoro in blocchi di dimensione s1×s2×s3 per migliorare la località di cache;
 * parallelizza i loop esterni con OpenMP. */
static void blocked_multiply(float *A, float *B_T, float *C, int a_rows, int a_cols, int b_cols){
	if (a_rows <= BLOCK_SIZE && a_cols <= BLOCK_SIZE && b_cols <= BLOCK_SIZE) {
		for (int i = 0; i < a_rows; i++)
			for (int j = 0; j < b_cols; j++) {
				float sum = 0.0f;
				for (int k = 0; k < a_cols; k++)
					sum += A[i * a_cols + k] * B_T[j * a_cols + k];
				C[i * b_cols + j] = sum;
			}
		return;
	}
	#pragma omp parallel for collapse(2) schedule(static)
	for (int i = 0; i < a_rows; i += BLOCK_SIZE)
		for (int j = 0; j < b_cols; j += BLOCK_SIZE)
			for (int k = 0; k < a_cols; k += BLOCK_SIZE)
				kernel(A, B_T, C, i, BLOCK_SIZE, j, BLOCK_SIZE, k, BLOCK_SIZE, a_rows, a_cols, b_cols);
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

	if (a_rows < 2 || b_rows < 2) {
		fprintf(stderr, "errore: @ richiede matrici 2D, non vettori (a[%d %d] b[%d %d])\n",
			a_rows, a_cols, b_rows, b_cols);
		instance_free(a);
		instance_free(b);
		return -1;
	}

	if (a_cols != b_rows){
		fprintf(stderr, "errore: shape incompatibili per prodotto matriciale a[%d %d] b[%d %d]\n",
			a_rows, a_cols, b_rows, b_cols);
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
	float total = 0.0f;
	#pragma omp parallel for reduction(+:total) schedule(static)
	for (int i = 0; i < n; i++)
		total += arr->data[i];
	instance_free(arr);
	float *sum = malloc(sizeof(float));
	if (sum == NULL) return -1;
	sum[0] = total;

	coppia shape;
	shape.row = 1;
	shape.col = 1;
	if (stack_push(my_stack, sum, shape) != 0) { free(sum); return -1; }
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
		instance_free(a);
		instance_free(b);
		return -1;
	}
	if (b->shape.row > 1) {
		fprintf (stderr, " errore: il secondo argomento non è un vettore");
		instance_free(a);
        instance_free(b);
		return -1;	
	}
	if (a->shape.col != b->shape.col) {
		fprintf(stderr, "errore: lunghezze incompatibili [%d] != [%d]\n", a->shape.col, b->shape.col);
		instance_free(a);
		instance_free(b);
		return -1;
	}
	int n = a->shape.col;
	float *sum = malloc(sizeof(float));
	if (sum == NULL) { instance_free(a); instance_free(b); return -1; }
	float dot = 0.0f;
	#pragma omp parallel for reduction(+:dot) schedule(static)
	for (int i = 0; i < n; i++)
		dot += a->data[i] * b->data[i];
	sum[0] = dot;

	instance_free(a);
	instance_free(b);

	coppia shape;
	shape.row = 1;
	shape.col = 1;
	if (stack_push(my_stack, sum, shape) != 0) { free(sum); return -1; }
	return 0;
}
float *padding(float *data, int row, int col, int pad_dim) {
    int row_padded = row + (2 * pad_dim);
    int col_padded = col + (2 * pad_dim);

    float *new_data = calloc((size_t)(row_padded * col_padded), sizeof(float));
    if (new_data == NULL) {
        return NULL;
    }

    for (int r = 0; r < row; r++) {
        memcpy(&new_data[(r + pad_dim) * col_padded + pad_dim],&data[r * col], sizeof(float) * (size_t)col);
    }

    return new_data;
}

float c_dot(float *window_start, float *k, int stride, int shape_k) {
    float sum = 0.0f;
    for (int r = 0; r < shape_k; r++) {
        for (int c = 0; c < shape_k; c++) {
            sum += window_start[r * stride + c] * k[r * shape_k + c];
        }
    }
    return sum;
}

int convoluzione(stack *my_stack) {
    array_instance *k = stack_pop(my_stack);
    if (!k)
        return -1;

    if (k->shape.row < 2 || k->shape.col < 2) {
        fprintf(stderr, "il kernel deve essere una matrice 2D\n");
        instance_free(k);
        return -1;
    }

    if (k->shape.col != k->shape.row || k->shape.col % 2 == 0) {
        fprintf(stderr, "il kernel deve essere una matrice quadrata di ordine dispari\n");
        instance_free(k);
        return -1;
    }

    array_instance *a = stack_pop(my_stack);
    if (!a) {
        instance_free(k);
        return -1;
    }

    int pad_dim = (k->shape.row - 1) / 2;
    float *expanded_data = padding(a->data, a->shape.row, a->shape.col, pad_dim);
    if (!expanded_data) {
        instance_free(k);
        instance_free(a);
        return -1;
    }

    int col_padded = a->shape.col + 2 * pad_dim;
    int shape_k = k->shape.row;
    int a_rows = a->shape.row;
    int a_cols = a->shape.col;

    float *out = malloc(sizeof(float) * (size_t)(a_rows * a_cols));
    if (!out) { free(expanded_data); instance_free(k); instance_free(a); return -1; }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            float *window = &expanded_data[i * col_padded + j];
            out[i * a_cols + j] = c_dot(window, k->data, col_padded, shape_k);
        }
    }

    free(expanded_data);
    instance_free(k);
    instance_free(a);

    coppia shape = {a_rows, a_cols};
    if (stack_push(my_stack, out, shape) != 0) { free(out); return -1; }
    return 0;
}



/* Parses a string literal starting after the opening '"' at s[offset].
 * Reads until the closing '"', allocates a copy and pushes it on the stack.
 * Input: s — full script string, offset — index of first char after '"', my_stack — destination stack.
 * Output: number of characters consumed (including closing '"'), or -1 on error. */
static long parse_string(const char *s, long offset, stack *my_stack) {
    int consumati = 0;
    char buf[256];
    sscanf(s + offset, "%255[^\"]%n", buf, &consumati);
    if (consumati == 0 || s[offset + consumati] != '"') {
        fprintf(stderr, "errore: stringa malformata\n");
        return -1;
    }
    if (stack_push_string(my_stack, buf) != 0) return -1;
    return 1 + consumati;
}

static void pgm_skip_comments (FILE *f){
  int c;
  while ((c = fgetc(f)) == '#') {
    while((c = fgetc(f)) != '\n' && c != EOF) {}
  }
  ungetc(c, f);
}
/* Pops a filename string, reads a PGM P5 grayscale image, normalises pixel
 * values to [0,1] dividing by 255, and pushes the result as a 2D tensor.
 * Input: my_stack — the stack (top: filename string).
 * Output: 0 on success, -1 on error. */
static int read_image(stack *my_stack){
  stack_item item = stack_pop_item(my_stack);
  if (item.type != ITEM_STRING){
    fprintf(stderr, "errore: expected string on top of stack\n");
    stack_free_item(item);
    return -1;
  }
  char *path = item.filename;
  FILE *f = fopen(path, "rb");
  if (!f){
    perror("fopen");
    free(path);
    return -1;
  }
  free(path);

  typedef struct { int width, height, maxval; } pgm_header;
  pgm_header h;
  char magic[3];
  pgm_skip_comments(f);
  if (fscanf(f, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '5'){
    fprintf(stderr, "read_image: formato non PGM P5\n");
    fclose(f);
    return -1;
  }
  pgm_skip_comments(f);
  if (fscanf(f, " %d", &h.width) != 1){ perror("fscanf"); fclose(f); return -1; }
  pgm_skip_comments(f);
  if (fscanf(f, " %d", &h.height) != 1){ perror("fscanf"); fclose(f); return -1; }
  pgm_skip_comments(f);
  if (fscanf(f, " %d", &h.maxval) != 1){ perror("fscanf"); fclose(f); return -1; }
  fgetc(f);

  size_t n = (size_t)h.width * (size_t)h.height;
  uint8_t *buf = malloc(n);
  if (!buf){ perror("malloc"); fclose(f); return -1; }
  if (fread(buf, 1, n, f) != n){ perror("fread"); free(buf); fclose(f); return -1; }

  float *data = malloc(sizeof(float) * n);
  if (!data){ perror("malloc"); free(buf); fclose(f); return -1; }
  for (size_t i = 0; i < n; i++)
    data[i] = (float)buf[i] / 255.0f;
  free(buf);
  fclose(f);

  coppia shape = { h.height, h.width };
  if (stack_push(my_stack, data, shape) != 0){ free(data); return -1; }
  return 0;
}

/* Pops a filename string and a tensor, writes it as a PGM (P5) grayscale image.
 * Values are clamped to [0, 1] then scaled to [0, 255] as uint8.
 * Input: my_stack — the stack (top: filename string, then tensor).
 * Output: 0 on success, -1 on error. */
int save_image(stack *my_stack) {
	stack_item item = stack_pop_item(my_stack);
	if (item.type != ITEM_STRING) {
		stack_free_item(item);
		return -1;
	}
	char *path = item.filename;

	array_instance *a = stack_pop(my_stack);
	if (!a) { free(path); return -1; }

	FILE *f = fopen(path, "wb");
	if (!f) { perror("fopen"); free(path); instance_free(a); return -1; }

	if (fprintf(f, "P5\n%d %d\n255\n", a->shape.col, a->shape.row) < 0) {
		perror("fprintf"); fclose(f); free(path); instance_free(a); return -1;
	}

	int n = a->shape.row * a->shape.col;
	uint8_t *buf = malloc((size_t)n);
	if (!buf) { fclose(f); free(path); instance_free(a); return -1; }
	for (int i = 0; i < n; i++) {
		float dato = a->data[i];
		float leq = (dato < 0.0f) ? 1.0f : 0.0f;
		float geq = (dato > 1.0f) ? 1.0f : 0.0f;
		dato = geq + dato * (1.0f - leq) * (1.0f - geq);
		buf[i] = (uint8_t)(dato * 255.0f);
	}

	if ((int)fwrite(buf, sizeof(uint8_t), (size_t)n, f) != n) {
		perror("fwrite"); free(buf); fclose(f); free(path); instance_free(a); return -1;
	}

	free(buf);
	fclose(f);
	free(path);
	instance_free(a);
	return 0;
}
/* Pops a shape tensor (1D, 1 or 2 elements) and pushes a new tensor of that shape
 * filled with random floats in [0, 1).
 * Validates dimensions >= 1 and product <= INT_MAX (entry point for user-supplied dims).
 * Input: my_stack — the stack (top: shape tensor [n] or [rows cols]).
 * Output: 0 on success, -1 on error. */
int random_array(stack *my_stack) {
	array_instance *s = stack_pop(my_stack);
	if (!s) return -1;
	int row, col;
	if (s->shape.row == 1 && s->shape.col == 2) {
		row = (int)s->data[0];
		col = (int)s->data[1];
	} else if (s->shape.row == 1 && s->shape.col == 1) {
		row = 1;
		col = (int)s->data[0];
	} else {
		fprintf(stderr, "errore: shape deve essere 1D di 1 o 2 elementi\n");
		instance_free(s);
		return -1;
	}
	if (row < 1 || col < 1) {
		fprintf(stderr, "errore: random_array richiede dimensioni >= 1\n");
		instance_free(s); return -1;
	}
	instance_free(s);
	int64_t n64 = (int64_t)row * (int64_t)col;
	if (n64 > INT_MAX) {
		fprintf(stderr, "errore: tensore troppo grande\n");
		return -1;
	}
	int n = (int)n64;
	float *arr = malloc(sizeof(float) * (size_t)n);
	if (!arr) { perror("malloc"); return -1; }
	for (int i = 0; i < n; i++)
		arr[i] = (float)rand() / (float)RAND_MAX;
	coppia shape = {row, col};
	if (stack_push(my_stack, arr, shape) != 0) { free(arr); return -1; }
	return 0;
}


/* Pops a tensor and pushes a new tensor with ReLU applied element-wise: max(0, x).
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int relu (stack *my_stack){
	array_instance *a = stack_pop(my_stack);
	if (!a) return -1;
	int n = a->shape.row * a->shape.col;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); return -1; }
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		new_data[i] = (a->data[i] > 0.f) ? a->data[i] : 0.f;
	coppia shape;
	shape.row = a->shape.row;
	shape.col = a->shape.col;
	int err = stack_push(my_stack, new_data,shape);
	if(err != 0){
		free(new_data);
		instance_free(a);
		return err;
	}
	instance_free(a);
	return 0;
}


/* Pops two tensors a (top) and b of identical shape, pushes element-wise min or max.
 * Input: my_stack — the stack, op — 'm' for min, 'M' for max.
 * Output: 0 on success, -1 on error. */
int extrema(stack *my_stack, char op){
	array_instance *a = stack_pop(my_stack);
	if (a == NULL) return -1;
	array_instance *b = stack_pop(my_stack);
	if (b == NULL) { instance_free(a); return -1; }
	if (a->shape.row != b->shape.row || a->shape.col != b->shape.col){
		fprintf(stderr, "errore: shape incompatibili [%d %d] != [%d %d]\n", a->shape.row, a->shape.col, b->shape.row, b->shape.col);
		instance_free(a);
		instance_free(b);
		return -1;
	}
	int row = a->shape.row;
	int col = a->shape.col;
	int n = row * col;
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { instance_free(a); instance_free(b); return -1; }
	/* Precedente: select aritmetico branchless x*cond + y*(1-cond).
	 * Rimosso perché inf*0 = NaN contamina il risultato anche nel ramo non scelto.
	 * fminf/fmaxf sono già branchless con -O3 e corretti su inf/NaN. */
	if (op == 'm'){
		for (int i = 0; i < n; i++)
			new_data[i] = fminf(a->data[i], b->data[i]);
	} else {
		for (int i = 0; i < n; i++)
			new_data[i] = fmaxf(a->data[i], b->data[i]);
	}
	instance_free(a);
	instance_free(b);
	coppia shape = {row, col};
	if (stack_push(my_stack, new_data, shape) != 0) { free(new_data); return -1; }
	return 0;
}


/* Swaps the top two stack items. Works for both tensors and strings.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_switch(stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return -1;
	stack_item b = stack_pop_item(my_stack);
	if (b.type == ITEM_NONE) {
		stack_free_item(a);
		return -1;
	}
	int err = stack_push_item(my_stack, a);
	stack_free_item(a);
	if (err != 0) { stack_free_item(b); return -1; }
	err = stack_push_item(my_stack, b);
	stack_free_item(b);
	return err;
}

/* Copies the second-from-top item to the top: ( a b -- a b a ).
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int over (stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return -1;
	stack_item b = stack_pop_item(my_stack);
	if (b.type == ITEM_NONE) {
		stack_free_item(a);
		return -1;
	}
	int err = stack_push_item(my_stack, b);
	if (err != 0) { 
		stack_free_item(b);
		stack_free_item(a);
		return -1;
	}
	err = stack_push_item(my_stack, a);
	stack_free_item(a);
	if (err != 0) { 
		stack_free_item(b);
		return -1;
	}
	err = stack_push_item(my_stack, b);
	stack_free_item(b);
	return err;	
}

/* Pops and discards the top stack item.
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int drop (stack *my_stack){
	stack_item a = stack_pop_item(my_stack);
	if (a.type == ITEM_NONE)
		return -1;
	stack_free_item(a);
	return 0;	
}

/* Flattens the top tensor to a 1D row vector in place: shape [r c] -> [1, r*c].
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int ravel(stack *my_stack) {
  array_instance *a = stack_pop(my_stack);
  if (!a) return -1;
  if (a->ref_count == 1){
    a->shape.col = a->shape.row * a->shape.col;
    a->shape.row = 1;
    if (stack_push_instance(my_stack, a) != 0) {
      instance_free(a);
      return -1;
    }
    instance_free(a);}
  else{
    int new_col = a->shape.row * a->shape.col;
    float *new_data = malloc(sizeof(float) * (size_t)new_col);
    memcpy(new_data, a->data, (size_t)new_col * sizeof(float));

    coppia shape;
    shape.row = 1;
    shape.col = new_col;

    instance_free(a);
    if (stack_push(my_stack, new_data, shape) != 0) {
      fprintf(stderr, "ravel: push_stack ha fallito");
      free(new_data);
      return -1;
    }
  }
	return 0;
}

/* Pops a tensor and pushes a 1D tensor [1×2] containing its shape [rows, cols].
 * Input: my_stack — the stack.
 * Output: 0 on success, -1 on error. */
int op_shape(stack *my_stack) {
	array_instance *a = stack_pop(my_stack);
	if (!a) return -1;
	int is_1d = (a->shape.row == 1);
	int count = is_1d ? 1 : 2;
	float *shape_arr = malloc(sizeof(float) * (size_t)count);
	if (!shape_arr) {
		perror("malloc");
		instance_free(a);
		return -1;
	}
	if (is_1d) {
		shape_arr[0] = (float)a->shape.col;
	} else {
		shape_arr[0] = (float)a->shape.row;
		shape_arr[1] = (float)a->shape.col;
	}
	instance_free(a);
	coppia shape = {1, count};
	if (stack_push(my_stack, shape_arr, shape) != 0) {
		free(shape_arr);
		return -1;
	}
	return 0;
}


/* Pops a value tensor v (top) and a shape tensor s, pushes a new tensor of shape s
 * filled by cycling through the elements of v.
 * Validates dimensions >= 1 and product <= INT_MAX (entry point for user-supplied dims).
 * Input: my_stack — the stack (top: v, then s).
 * Output: 0 on success, -1 on error. */
int fill(stack *my_stack) {
	array_instance *v = stack_pop(my_stack);
	if (!v) return -1;
	array_instance *s = stack_pop(my_stack);
	if (!s) { instance_free(v); return -1; }

	if (s->shape.row != 1 || s->shape.col < 1 || s->shape.col > 2) {
		fprintf(stderr, "errore: fill richiede shape 1D di 1 o 2 elementi\n");
		instance_free(s); instance_free(v); return -1;
	}
	int row = (s->shape.col == 2) ? (int)s->data[0] : 1;
	int col = (s->shape.col == 2) ? (int)s->data[1] : (int)s->data[0];
	if (row < 1 || col < 1) {
		fprintf(stderr, "errore: fill richiede dimensioni >= 1\n");
		instance_free(s); instance_free(v); return -1;
	}
	instance_free(s);

	int64_t n64 = (int64_t)row * (int64_t)col;
	if (n64 > INT_MAX) {
		fprintf(stderr, "errore: tensore troppo grande\n");
		instance_free(v); return -1;
	}
	int n = (int)n64;
	int m = v->shape.row * v->shape.col;
	if (m <= 0) {
		fprintf(stderr, "fill: tensore valore non può essere vuoto\n");
		instance_free(v);
		return -1;
	}
	float *new_data = malloc(sizeof(float) * (size_t)n);
	if (!new_data) { perror("malloc"); instance_free(v); return -1; }

	for (int i = 0; i < n; i++)
		new_data[i] = v->data[i % m];

	instance_free(v);
	coppia shape = {row, col};
	if (stack_push(my_stack, new_data, shape) != 0) {
		free(new_data);
		return -1;
	}
	return 0;
}


/* Pops a filename string and a tensor, serialises the tensor to a binary file.
 * Header (64 bytes): shape, ndim, data_offset. Float data starts at byte 64.
 * Input: my_stack — the stack (top: filename string, then tensor).
 * Output: 0 on success, -1 on argument error, -2 on I/O error. */
int on_disk_save(stack *my_stack){
	stack_item item = stack_pop_item(my_stack);
	if (item.type != ITEM_STRING) {
		fprintf(stderr, "errore: atteso un filename");
		return -1;
	}
	char *path = item.filename;
	array_instance *a = stack_pop(my_stack);
	if (!a){
		free(path);
		return -1;
	}

	disk_header header = {0};
	header.shape[0] = a->shape.row;
	header.shape[1] = a->shape.col;
	if (header.shape[0] == 1)
		header.ndim = 1;
	else
		header.ndim = 2;
	header.data_offset = 64;

	FILE *fd = fopen(path, "wb");
	if(!fd){
		perror("fopen");
		free(path);
		instance_free(a);
		return -2;
	}
	if(fwrite(&header, sizeof(header), 1, fd) != 1){
		perror("fwrite");
		fclose(fd);
		free(path);
		instance_free(a);
		return -2;
	}
	if(fseek(fd, header.data_offset, SEEK_SET) != 0){
		perror("fseek");
		fclose(fd);
		free(path);
		instance_free(a);
		return -2;
	}
	size_t n = (size_t)(a->shape.row * a->shape.col);
	if(fwrite(a->data, sizeof(float), n, fd) != n){
		perror("fwrite");
		fclose(fd);
		free(path);
		instance_free(a);
		return -2;
	}

	fclose(fd);
	free(path);
	instance_free(a);
	return 0;
	
	
}


/* Pops a filename string, maps the binary tensor file into memory (read-only, no copy),
 * and pushes the result on the stack. instance_free will munmap using on_disk and data_offset.
 * Input: my_stack — the stack (top: filename string).
 * Output: 0 on success, -1 on error. */
int on_disk_read(stack *my_stack) {
	stack_item item = stack_pop_item(my_stack);
	if (item.type != ITEM_STRING) {
		fprintf(stderr, "errore: atteso un filename\n");
		return -1;
	}
	char *path = item.filename;

	int f = open(path, O_RDONLY);
	if (f < 0) { perror("open"); free(path); return -1; }

	struct stat st;
	if (fstat(f, &st) < 0) { perror("fstat"); close(f); free(path); return -1; }

	disk_header hdr;
	if (read(f, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
		perror("read"); close(f); free(path); return -1;
	}

	void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, f, 0);
	close(f);
	if (map == MAP_FAILED) { perror("mmap"); free(path); return -1; }

	float *src = (float *)((char *)map + hdr.data_offset);
	coppia shape = { hdr.shape[0], hdr.shape[1] };
	array_instance *inst = new_instance(src, shape);
	if (!inst) { munmap(map, (size_t)st.st_size); free(path); return -1; }
	inst->on_disk = 1;
	inst->data_offset = (int32_t)hdr.data_offset;

	if (stack_push_instance(my_stack, inst) != 0) {
		instance_free(inst);
		free(path);
		return -1;
	}
	instance_free(inst);
	free(path);
	return 0;
}

/* Main interpreter loop. Scans s token by token and dispatches each command.
 * Enforces exactly one whitespace separator between consecutive tokens.
 * Input: s — null-terminated script string, my_stack — the execution stack.
 * Output: 0 on success, -1 on error. */
int parser(const char *s, stack *my_stack){
	long size = (long)strlen(s);

	/* strip trailing whitespace so trailing newlines are never an error */
	while (size > 0 && (s[size-1] == ' ' || s[size-1] == '\n' ||
	                    s[size-1] == '\t' || s[size-1] == '\r'))
		size--;

	/* BEFORE_FIRST=0: leading spaces OK; NEED_SEP=1: must see one space;
	 * SEP_DONE=2: space consumed, next char must be a token */
	int sep_state = 0;

	for (long i = 0; i < size; i++){
		int is_ws = (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r');

		if (is_ws) {
			if      (sep_state == 1) sep_state = 2;
			else if (sep_state == 2) {
				fprintf(stderr, "errore: spazio extra alla posizione %ld\n", i);
				return -1;
			}
			continue;
		}

		if (sep_state == 1) {
			fprintf(stderr, "errore: manca spazio tra i token alla posizione %ld\n", i);
			return -1;
		}

		char token[2] = {s[i], '\0'};
 		switch(lookup(token)){

			case OP_NEW_ARRAY:{
				long result = parse_array(s, i+1, my_stack);
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
			case OP_READ_NAME:{
				long result = parse_string(s, i+1, my_stack);
				if (result == -1) return -1;
					i += result;
				break;}

			case OP_LOAD_PGM: if (read_image(my_stack) != 0) return -1; break;
			case OP_SAVE_PGM: if (save_image(my_stack) != 0) return -1; break;
			case OP_RANDOM: if (random_array(my_stack) != 0) return -1; break;
			case OP_RELU: if (relu(my_stack) != 0) return -1; break;
			case OP_MIN: if (extrema(my_stack, 'm') != 0) return -1; break;
			case OP_MAX: if (extrema(my_stack, 'M') != 0) return -1; break;
			case OP_SWITCH: if (op_switch(my_stack) != 0) return -1; break;
			case OP_OVER: if (over(my_stack) != 0) return -1; break;
			case OP_DROP: if (drop(my_stack) != 0) return -1; break;
			case OP_RAVEL: if (ravel(my_stack) != 0) return -1; break;
			case OP_SHAPE: if (op_shape(my_stack) != 0) return -1; break;
			case OP_FILL: if (fill(my_stack) != 0) return -1; break;
			case OP_SAVE_DISK: if (on_disk_save(my_stack) != 0) return -1; break;
			case OP_LOAD_DISK: if (on_disk_read(my_stack) != 0) return -1; break;
			case OP_CONV: if (convoluzione(my_stack) != 0) return -1; break;

			default:
				fprintf(stderr, "errore comando sconosciuto: '%c'\n", s[i]);
				return -1;
		}
		sep_state = 1;
	}

	return 0;
}
