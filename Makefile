CC = gcc
CFLAGS = -std=c99 -O3 -Wall
LIBS = -lm -lfftw3f
INCL = -I. -Iio -Ilib -Imath


# Settings for FFTW
FFTW_DIR = /opt/cray/pe/fftw/3.3.8.4/$(CRAY_CPU_TARGET)
ifneq ($(FFTW_DIR),)
  LIBS += -L$(FFTW_DIR)/lib
  INCL += -I$(FFTW_DIR)/include
endif

# Setting for single precision density fields and FFT
LIBS += -DSINGLE_PREC

# Settings for OpenMP (comment the following line to disable OpenMP)
LIBS += -DOMP -fopenmp -lfftw3f_omp

# Settings for CFITSIO (not implemented yet)

SRCS = $(wildcard *.c lib/*.c io/*.c math/*.c)
EXEC = libpowspec.so

all:
	$(CC) $(CFLAGS) -fPIC -shared -o $(EXEC) $(SRCS) $(LIBS) $(INCL)

clean:
	rm $(EXEC)
