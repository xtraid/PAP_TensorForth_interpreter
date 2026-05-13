CC = gcc
HEADERS = ${wildcard *.h}
SRC = ${wildcard *.c}
OBJ = ${SRC:.c=.o}
DEBUG = F
PERF = F

BASE_FLAGS = -std=c17 -Wall -Wextra -pedantic -Wshadow -Wconversion -Wstrict-prototypes -fopenmp #most annoying warning possible so i can make it right

ifeq "${DEBUG}" "TRUE"
	CFLAGS = -O0 -g ${BASE_FLAGS}
else ifeq "${PERF}" "TRUE"
	CFLAGS = -O3 -march=native -ffast-math -funroll-loops ${BASE_FLAGS}
else
	CFLAGS = -O3 -funroll-loops ${BASE_FLAGS}
endif

all:TensorForth

%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $<

TensorForth: ${OBJ}
	${CC} ${CFLAGS} -o $@ $^
	
.PHONY: all clean

clean:
	rm -f ${OBJ}
	rm -f TensorForth