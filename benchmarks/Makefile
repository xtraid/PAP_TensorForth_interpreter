CC = gcc
BASE = -std=c17 -Wall -Wextra -Wshadow -Wconversion -fopenmp
TF_CFLAGS ?= -O3 -march=native -ffast-math -funroll-loops
# Set OPENBLAS_PREFIX for an unpacked/local installation; otherwise use pkg-config.
ifdef OPENBLAS_PREFIX
BLAS_CFLAGS = -I$(OPENBLAS_PREFIX)/include/openblas
BLAS_LIBS = -L$(OPENBLAS_PREFIX)/lib -Wl,-rpath,$(abspath $(OPENBLAS_PREFIX)/lib) -lopenblas
else
BLAS_CFLAGS = $(shell pkg-config --cflags openblas)
BLAS_LIBS = $(shell pkg-config --libs openblas)
endif
SOURCES = $(filter-out ../main.c,$(wildcard ../*.c))
OBJECTS = $(patsubst ../%.c,build/%.o,$(SOURCES))

.PHONY: all
all: build/matmul_bench

build:
	mkdir -p build

build/%.o: ../%.c $(wildcard ../*.h) Makefile | build
	$(CC) $(BASE) $(TF_CFLAGS) -I.. -c $< -o $@

# Keep validation IEEE-compliant even when the runtime uses fast-math.
build/matmul_bench.o: matmul_bench.c $(wildcard ../*.h) Makefile | build
	$(CC) $(BASE) -O3 -I.. $(BLAS_CFLAGS) -c $< -o $@

build/matmul_bench: build/matmul_bench.o $(OBJECTS)
	$(CC) $(BASE) $^ $(BLAS_LIBS) -lm -o $@
