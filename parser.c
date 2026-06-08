#include "parser.h"
#include <ctype.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/* forward declarations for functions not listed in parser.h */
int save_image(stack *my_stack);
int read_image(stack *my_stack);
int convoluzione(stack *my_stack);

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

	/* prima passata: valida la sintassi e conta gli elementi.
	 * Gli elementi sono separati da uno o più spazi: dopo '[' e dopo ogni valore
	 * si saltano tutti gli spazi consecutivi (spec: separatore = spazi). */
	if (s[tmp] != ' ') {
		fprintf(stderr, "error: missing space after '['\n");
		return -1;
	}
	while (s[tmp] == ' ') tmp++;

	while (s[tmp] != '\0' && s[tmp] != ']') {
		float val;
		if (sscanf(s + tmp, "%f%n", &val, &consumati) == 1) {
			if (s[tmp + consumati] != ' ') {
				if (s[tmp + consumati] == ']')
					fprintf(stderr, "error: missing space before ']'\n");
				else
					fprintf(stderr, "error: missing space after value %g\n", val);
				return -1;
			}
			count++;
			tmp += consumati;
			while (s[tmp] == ' ') tmp++;
		} else {
			if (isprint((unsigned char)s[tmp]))
				fprintf(stderr, "error: invalid token '%c' at position %ld\n", s[tmp], tmp - start);
			else
				fprintf(stderr, "error: invalid byte '\\x%02x' at position %ld\n", (unsigned char)s[tmp], tmp - start);
			return -1;
		}
	}

	if (s[tmp] == '\0') {
		fprintf(stderr, "error: missing closing ']'\n");
		return -1;
	}

	long end = tmp;  /* posizione di ']' */

	if (count == 0) {
		fprintf(stderr, "error: empty array '[ ]' is not supported\n");
		return -1;
	}

	/* seconda passata: alloca esattamente `count` elementi e riempie.
	 * Usa la stessa scansione della prima passata (skip degli spazi multipli)
	 * per restare allineata sul numero di elementi contati. */
	float *new_array = malloc(sizeof(float) * (size_t)count);
	if (new_array == NULL) return -1;

	tmp = offset;
	while (s[tmp] == ' ') tmp++;  /* salta gli spazi dopo '[' */
	for (int j = 0; j < count; j++) {
		float val;
		if (sscanf(s + tmp, "%f%n", &val, &consumati) == 1) {
			new_array[j] = val;
			tmp += consumati;
			while (s[tmp] == ' ') tmp++;
		} else {
			free(new_array);
			return -1;
		}
	}

	shape_t shape;
	shape.row = 1;
	shape.col = count;
  shape.ndim = 1;

	if (stack_push(my_stack, new_array, shape) != 0) {
		free(new_array);
		return -1;
	}
	return end - start + 1;
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
        fprintf(stderr, "error: malformed string literal\n");
        return -1;
    }
    if (stack_push_string(my_stack, buf) != 0) return -1;
    return 1 + consumati;
}

/* Main interpreter loop. Scans s token by token and dispatches each command.
 * Enforces exactly one whitespace separator between consecutive tokens.
 * Input: s — null-terminated script string, my_stack — the execution stack.
 * Output: TF_OK on success, TFError code on error. */
int parser(const char *s, stack *my_stack){
	long size = (long)strlen(s);

	/* strip trailing whitespace so trailing newlines are never an error */
	while (size > 0 && (s[size-1] == ' ' || s[size-1] == '\n' ||
	                    s[size-1] == '\t' || s[size-1] == '\r'))
		size--;

	/* BEFORE_FIRST=0: leading spaces OK; NEED_SEP=1: must see one space; */
	
	int sep_state = 0;

	for (long i = 0; i < size; i++){
		int is_ws = (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r');

		if (is_ws) {
			if (sep_state == 1) sep_state = 2;
      continue;
		}

		if (sep_state == 1) {
			fprintf(stderr, "error: missing space between tokens at position %ld\n", i);
			return TF_ERR_SYNTAX;
		}

		char token[2] = {s[i], '\0'};
		switch(lookup(token)){

			case OP_NEW_ARRAY:{
				long result = parse_array(s, i+1, my_stack);
				if (result == -1) return TF_ERR_SYNTAX;
				i += result;
				break;}
			case OP_READ_NAME:{
				long result = parse_string(s, i+1, my_stack);
				if (result == -1) return TF_ERR_SYNTAX;
				i += result;
				break;}

			case OP_PRINT_N_POP:  { int e = pop_print(my_stack);              if (e) return e; break; }
			case OP_PRINT_MATRIX: { int e = pop_print_as_matrix(my_stack);    if (e) return e; break; }
			case OP_DUPLICATE:    { int e = duplicate(my_stack);               if (e) return e; break; }
			case OP_ADD:          { int e = algebra(my_stack, 'a');            if (e) return e; break; }
			case OP_SUBTRACTION:  { int e = algebra(my_stack, 's');            if (e) return e; break; }
			case OP_PRODUCT:      { int e = algebra(my_stack, 'p');            if (e) return e; break; }
			case OP_GREATER:      { int e = comparisons(my_stack, 'M');     if (e) return e; break; }
			case OP_LESSER:       { int e = comparisons(my_stack, 'm');     if (e) return e; break; }
			case OP_CONFRONT:     { int e = comparisons(my_stack, 'u');     if (e) return e; break; }
			case OP_AND:          { int e = logical_binary(my_stack, 'a');   if (e) return e; break; }
			case OP_OR:           { int e = logical_binary(my_stack, 'o');   if (e) return e; break; }
			case OP_NOT:          { int e = op_not(my_stack);                  if (e) return e; break; }
			case OP_MASK:         { int e = mask(my_stack);                    if (e) return e; break; }
			case OP_MAT_MUL:      { int e = mat_mat_mul(my_stack);             if (e) return e; break; }
			case OP_RESHAPE:      { int e = op_reshape(my_stack);              if (e) return e; break; }
			case OP_SUM:          { int e = sum_arr(my_stack);                 if (e) return e; break; }
			case OP_DOT:          { int e = dot_product(my_stack);             if (e) return e; break; }
			case OP_LOAD_PGM:     { int e = read_image(my_stack);              if (e) return e; break; }
			case OP_SAVE_PGM:     { int e = save_image(my_stack);              if (e) return e; break; }
			case OP_RANDOM:       { int e = random_array(my_stack);            if (e) return e; break; }
			case OP_RELU:         { int e = relu(my_stack);                    if (e) return e; break; }
			case OP_MIN:          { int e = extrema(my_stack, 'm');            if (e) return e; break; }
			case OP_MAX:          { int e = extrema(my_stack, 'M');            if (e) return e; break; }
			case OP_SWITCH:       { int e = op_switch(my_stack);               if (e) return e; break; }
			case OP_OVER:         { int e = over(my_stack);                    if (e) return e; break; }
			case OP_DROP:         { int e = drop(my_stack);                    if (e) return e; break; }
			case OP_RAVEL:        { int e = ravel(my_stack);                   if (e) return e; break; }
			case OP_SHAPE:        { int e = op_shape(my_stack);                if (e) return e; break; }
			case OP_FILL:         { int e = fill(my_stack);                    if (e) return e; break; }
			case OP_SAVE_DISK:    { int e = on_disk_save(my_stack);            if (e) return e; break; }
			case OP_LOAD_DISK:    { int e = on_disk_read(my_stack);            if (e) return e; break; }
			case OP_CONV:         { int e = convoluzione(my_stack);            if (e) return e; break; }

			default:
				fprintf(stderr, "error: unknown command '%c'\n", s[i]);
				return TF_ERR_SYNTAX;
		}
		sep_state = 1;
	}

	return TF_OK;
}
