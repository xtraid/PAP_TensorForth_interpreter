#ifndef READ_FILE
#define READ_FILE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

FILE* open_file (char *nome);

char* get_command (FILE *);

#endif