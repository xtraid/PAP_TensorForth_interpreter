#include "parser.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

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
int read_image(stack *my_stack){
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

