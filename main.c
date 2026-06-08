#include "reader.h"
#include "parser.h"
#include <stdlib.h>
#include <time.h>

static const char *tf_category (int e){
  switch (e) {
    case TF_ERR_STACK: return "stack error";
    case TF_ERR_SHAPE: return "shape mismatch";
    case TF_ERR_TYPE: return "type error";
    case TF_ERR_ARG: return "invalid argument";
    case TF_ERR_IO: return "I/O error";
    case TF_ERR_MEM: return "memory error";
    case TF_ERR_SYNTAX: return "syntax error";
    default: return "error";
  }
}

int main (int argc, char *argv[]){
	srand((unsigned int)time(NULL));
	if (argc < 2){
		fprintf(stderr, "usage: tensorForth <script>\n");
		return 1;
	}
	FILE *f = open_file(argv[1]);
	if (f == NULL){
		return 1;		
	}
	char *script = get_command(f);
	fclose(f);
	stack *my_stack = stack_init();
	int err = parser(script, my_stack);
  free(script);
  stack_free(my_stack);
  if (err != TF_OK)
    fprintf(stderr, "abort: %s\n", tf_category(err));
  return err != TF_OK ? 1 : 0;	
}
