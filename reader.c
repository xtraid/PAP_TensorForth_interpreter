#include "reader.h"

FILE* open_file (char *nome){
	FILE *fd = fopen(nome, "r");
	if (fd == NULL){
		perror("fopen");
		return NULL;
	}
	return fd;
}

char* get_command (FILE *fd){
	int err = fseek(fd, 0, SEEK_END);
	if(err != 0){
		perror("fseek");
		return NULL;
	}
	long size = ftell(fd);
	rewind(fd);
	char *string = malloc(sizeof(char)*(size+1));
	for (int i = 0; i<size; i++){
		string[i]=fgetc(fd);
		
	}
	string[size] = '\0';
	return string;
}


