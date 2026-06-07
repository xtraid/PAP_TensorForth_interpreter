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
