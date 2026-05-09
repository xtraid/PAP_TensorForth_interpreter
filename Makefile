CC = gcc
HEADERS = ${wildcard *.h}
SRC = ${wildcard *.c}
OBJ = ${SRC:.c=.o}
DEBUG = F
ifeq "${DEBUG}" "TRUE"
	CFLAGS = -O0 -g -std=c17 -Wall -Wextra -pedantic -Wshadow -Wconversion -Wstrict-prototypes -fopenmp #most annoying warning possible so i can make it right
else
	CFLAGS = -O3 -std=c17 -Wall -Wextra -pedantic -Wshadow -Wconversion -Wstrict-prototypes -fopenmp #most annoying warning possible so i can make it right
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