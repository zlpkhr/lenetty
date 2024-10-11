# Makefile for the dumb
#
CC = gcc
CFLAGS = -O3
CC = gcc
CFLAGS = -O3
CXX = g++
CXXFLAGS = -O3 -Wall -DFPNUM=$(DEFFP)
GUNZIP = gzip -d
DEFFP ?= float

%.c: %.c.gz
	$(GUNZIP) -c $< > $@

all : int-lenet float-lenet floatx-lenet

int-lenet: int-lenet.o int8_t_images.o
float-lenet: float-lenet.o float_images.o
floatx-lenet: floatx-lenet.o float_images.o
	$(CXX) -o $@ $^

int8_t_images.c : int8_t_images.c.gz
float_images.c: float_images.c.gz

int8_t_images.c : dump-images.py
float_images.c : dump-images.py

int8_t_parameters.c : dump-parameters.py
float_parameters.c : dump-parameters.py
	
clean : 
	$(RM) int-lenet float-lenet *.o
