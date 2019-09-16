# makefile overrides to use Intel ICC compiler, double prec only.

CC=icc
FC=ifort

FFLAGS=-fPIC -O3 -xHost
CFLAGS=-std=c99 -fPIC -O3 -xHost

OMPFLAGS=-qopenmp
OMPLIBS=-qopenmp

CLINK=-lifcore -qopenmp
LIBS=-lifcore -qopenmp
