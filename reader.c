#include "reader.h"

/* Opens a file in read mode.
 * Input: nome — path to the file.
 * Output: FILE* on success, NULL on failure. */
FILE* open_file (char *nome){
	FILE *fd = fopen(nome, "r");
	if (fd == NULL){
		perror("fopen");
		return NULL;
	}
	return fd;
}

/* Reads the entire content of fd into a null-terminated string.
 * Input: fd — open file descriptor.
 * Output: heap-allocated string with file content, caller must free it. NULL on error. */
char* get_command (FILE *fd){
	int err = fseek(fd, 0, SEEK_END);
	if(err != 0){
		perror("fseek");
		return NULL;
	}
	long size = ftell(fd);
	if (size < 0) { 
		perror("ftell");
		return NULL; 
		}
	rewind(fd);
	char *string = malloc(sizeof(char)*(size_t)(size+1));
	for (long i = 0; i < size; i++){
		int c = fgetc(fd);
		if (c == EOF) {
			string[i] = '\0';
			return string;
		}
		string[i] = (char)c;
	}
	string[size] = '\0';
	return string;
}


