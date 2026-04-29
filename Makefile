CC = gcc
HEADERS = ${wildcard *.h}
SRC = ${wildcard *.c}
OBJ = ${SRC:.c=.o}
DEBUG = TRUE
ifeq "${DEBUG}" "TRUE"
	CFLAGS = -O0 -g -std=C17 -Wall -Wextra -pedantic -Wshadow -Wconversion -Wstrict-prototypes #most annoying warning possible so i can make it right 
else
	CFLAGS = -O3 -std=C17 -Wall -Wextra -pedantic -Wshadow -Wconversion -Wstrict-prototypes #most annoying warning possible so i can make it right 
endif

all:TensorFort

%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $<

TensorFort: ${OBJ}
	${CC} ${CFLAGS} -o $@ $^
	
.PHONY: all clean

clean:
	rm -f ${OBJ}
	rm -f TensorFort