#include "reader.h"
#include "parser.h"
#include <stdlib.h>
#include <time.h>

int main (int argc, char *argv[]){
	srand((unsigned int)time(NULL));
	if (argc < 2){
		fprintf(stderr, "usage: TensorForth <script>\n");
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
	return err;
	
}
