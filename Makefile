# Makefile for the dumb
#
CC = gcc
CFLAGS = -O3
GUNZIP = gzip -d

# Ensure that .c files are decompressed if needed
%.c: %.c.gz
	$(GUNZIP) -c $< > $@

# Target executable
all : int-lenet

int-lenet: int-lenet.o int8_t_images.o
	$(CC) $(CFLAGS) int-lenet.o int8_t_images.o -o int-lenet

# Individual object file rules
int-lenet.o: int-lenet.c
	$(CC) $(CFLAGS) -c int-lenet.c

int8_t_images.o: int8_t_images.c
	$(CC) $(CFLAGS) -c int8_t_images.c

# Rules to decompress sources if needed
int8_t_images.c : int8_t_images.c.gz

int8_t_images.c : dump-images.py

int8_t_parameters.c : dump-parameters.py

# Clean up
clean : 
	$(RM) int-lenet *.o
