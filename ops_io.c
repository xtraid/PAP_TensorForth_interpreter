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
 * Output: TF_OK on success. */
int read_image(stack *my_stack){
  stack_item item = stack_pop_item(my_stack);
  if (item.type != ITEM_STRING){
    fprintf(stderr, "error: expected filename string on top of stack\n");
    stack_free_item(item);
    return TF_ERR_STACK;
  }
  char *path = item.filename;
  FILE *f = fopen(path, "rb");
  if (!f){
    perror("fopen");
    free(path);
    return TF_ERR_IO;
  }
  free(path);

  typedef struct { int width, height, maxval; } pgm_header;
  pgm_header h;
  char magic[3];
  pgm_skip_comments(f);
  if (fscanf(f, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '5'){
    fprintf(stderr, "error: '(': not a valid PGM P5 file\n");
    fclose(f);
    return TF_ERR_IO;
  }
  pgm_skip_comments(f);
  if (fscanf(f, " %d", &h.width) != 1){ perror("fscanf"); fclose(f); return TF_ERR_IO; }
  pgm_skip_comments(f);
  if (fscanf(f, " %d", &h.height) != 1){ perror("fscanf"); fclose(f); return TF_ERR_IO; }
  pgm_skip_comments(f);
  if (fscanf(f, " %d", &h.maxval) != 1){ perror("fscanf"); fclose(f); return TF_ERR_IO; }
  fgetc(f);

  size_t n = (size_t)h.width * (size_t)h.height;
  uint8_t *buf = malloc(n);
  if (!buf){ perror("malloc"); fclose(f); return TF_ERR_MEM; }
  if (fread(buf, 1, n, f) != n){ perror("fread"); free(buf); fclose(f); return TF_ERR_IO; }

  float *data = malloc(sizeof(float) * n);
  if (!data){ perror("malloc"); free(buf); fclose(f); return TF_ERR_MEM; }
  for (size_t i = 0; i < n; i++)
    data[i] = (float)buf[i] / 255.0f;
  free(buf);
  fclose(f);

  shape_t shape = { h.height, h.width, 2};
  if (stack_push(my_stack, data, shape) != 0){ free(data); return TF_ERR_MEM; }
  return TF_OK;
}

/* Pops a filename string and a tensor, writes it as a PGM (P5) grayscale image.
 * Values are clamped to [0, 1] then scaled to [0, 255] as uint8.
 * Input: my_stack — the stack (top: filename string, then tensor).
 * Output: TF_OK on success. */
int save_image(stack *my_stack) {
	stack_item item = stack_pop_item(my_stack);
	if (item.type != ITEM_STRING) {
		stack_free_item(item);
		return TF_ERR_STACK;
	}
	char *path = item.filename;

	array_instance *a = stack_pop(my_stack);
	if (!a) { free(path); return TF_ERR_STACK; }

	FILE *f = fopen(path, "wb");
	if (!f) { perror("fopen"); free(path); instance_free(a); return TF_ERR_IO; }

	if (fprintf(f, "P5\n%d %d\n255\n", a->shape.col, a->shape.row) < 0) {
		perror("fprintf"); fclose(f); free(path); instance_free(a); return TF_ERR_IO;
	}

	int n = a->shape.row * a->shape.col;
	uint8_t *buf = malloc((size_t)n);
	if (!buf) { fclose(f); free(path); instance_free(a); return TF_ERR_MEM; }
	for (int i = 0; i < n; i++) {
		float dato = a->data[i];
		float leq = (dato < 0.0f) ? 1.0f : 0.0f;
		float geq = (dato > 1.0f) ? 1.0f : 0.0f;
		dato = geq + dato * (1.0f - leq) * (1.0f - geq);
		buf[i] = (uint8_t)(dato * 255.0f);
	}

	if ((int)fwrite(buf, sizeof(uint8_t), (size_t)n, f) != n) {
		perror("fwrite"); free(buf); fclose(f); free(path); instance_free(a); return TF_ERR_IO;
	}

	free(buf);
	fclose(f);
	free(path);
	instance_free(a);
	return TF_OK;
}

/* Pops a shape tensor (1D, 1 or 2 elements) and pushes a new tensor of that shape
 * filled with random floats in [0, 1).
 * Validates dimensions >= 1 and product <= INT_MAX (entry point for user-supplied dims).
 * Input: my_stack — the stack (top: shape tensor [n] or [rows cols]).
 * Output: TF_OK on success. */
int random_array(stack *my_stack) {
	array_instance *s = stack_pop(my_stack);
	if (!s) return TF_ERR_STACK;
	int row, col;
	if (s->shape.row == 1 && s->shape.col == 2) {
		row = (int)s->data[0];
		col = (int)s->data[1];
	} else if (s->shape.row == 1 && s->shape.col == 1) {
		row = 1;
		col = (int)s->data[0];
	} else {
		fprintf(stderr, "error: '?' requires a 1D shape tensor with 1 or 2 elements\n");
		instance_free(s);
		return TF_ERR_ARG;
	}
	if (row < 1 || col < 1) {
		fprintf(stderr, "error: '?' requires dimensions >= 1\n");
		instance_free(s); return TF_ERR_ARG;
	}
	int ndim = s->shape.col;
	instance_free(s);
	int64_t n64 = (int64_t)row * (int64_t)col;
	if (n64 > INT_MAX) {
		fprintf(stderr, "error: tensor too large\n");
		return TF_ERR_ARG;
	}
	int n = (int)n64;
	float *arr = malloc(sizeof(float) * (size_t)n);
	if (!arr) { perror("malloc"); return TF_ERR_MEM; }
	for (int i = 0; i < n; i++)
		arr[i] = (float)rand() / (float)RAND_MAX;
	shape_t shape = {row, col, ndim};
	if (stack_push(my_stack, arr, shape) != 0) { free(arr); return TF_ERR_MEM; }
	return TF_OK;
}


/* Pops a filename string and a tensor, serialises the tensor to a binary file.
 * Header (64 bytes): shape, ndim, data_offset. Float data starts at byte 64.
 * Input: my_stack — the stack (top: filename string, then tensor).
 * Output: TF_OK on success. */
int on_disk_save(stack *my_stack){
	stack_item item = stack_pop_item(my_stack);
	if (item.type != ITEM_STRING) {
		fprintf(stderr, "error: expected filename string on top of stack\n");
		return TF_ERR_STACK;
	}
	char *path = item.filename;
	array_instance *a = stack_pop(my_stack);
	if (!a){
		free(path);
		return TF_ERR_STACK;
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
		return TF_ERR_IO;
	}
	if(fwrite(&header, sizeof(header), 1, fd) != 1){
		perror("fwrite");
		fclose(fd);
		free(path);
		instance_free(a);
		return TF_ERR_IO;
	}
	if(fseek(fd, header.data_offset, SEEK_SET) != 0){
		perror("fseek");
		fclose(fd);
		free(path);
		instance_free(a);
		return TF_ERR_IO;
	}
	size_t n = (size_t)(a->shape.row * a->shape.col);
	if(fwrite(a->data, sizeof(float), n, fd) != n){
		perror("fwrite");
		fclose(fd);
		free(path);
		instance_free(a);
		return TF_ERR_IO;
	}

	fclose(fd);
	free(path);
	instance_free(a);
	return TF_OK;
}


/* Pops a filename string, maps the binary tensor file into memory (read-only, no copy),
 * and pushes the result on the stack. instance_free will munmap using on_disk and data_offset.
 * Input: my_stack — the stack (top: filename string).
 * Output: TF_OK on success. */
int on_disk_read(stack *my_stack) {
	stack_item item = stack_pop_item(my_stack);
	if (item.type != ITEM_STRING) {
		fprintf(stderr, "error: expected filename string on top of stack\n");
		return TF_ERR_STACK;
	}
	char *path = item.filename;

	int f = open(path, O_RDONLY);
	if (f < 0) { perror("open"); free(path); return TF_ERR_IO; }

	struct stat st;
	if (fstat(f, &st) < 0) { perror("fstat"); close(f); free(path); return TF_ERR_IO; }

	disk_header hdr;
	if (read(f, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
		perror("read"); close(f); free(path); return TF_ERR_IO;
	}

	void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, f, 0);
	close(f);
	if (map == MAP_FAILED) { perror("mmap"); free(path); return TF_ERR_IO; }

	float *src = (float *)((char *)map + hdr.data_offset);
	shape_t shape = { hdr.shape[0], hdr.shape[1], hdr.ndim };
	array_instance *inst = new_instance(src, shape);
	if (!inst) { munmap(map, (size_t)st.st_size); free(path); return TF_ERR_MEM; }
	inst->on_disk = 1;
	inst->data_offset = (int32_t)hdr.data_offset;

	if (stack_push_instance(my_stack, inst) != 0) {
		instance_free(inst);
		free(path);
		return TF_ERR_MEM;
	}
	instance_free(inst);
	free(path);
	return TF_OK;
}
