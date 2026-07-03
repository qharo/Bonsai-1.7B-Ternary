CC = clang
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS_BASE = -O3 -std=c11 -Wall -fPIC -march=native -ffast-math
    FRAMEWORKS = -framework Accelerate
else
    # Linux / HF Spaces: cap at x86-64-v2 to avoid AVX-512 SIGILL on runtime nodes
    CFLAGS_BASE = -O3 -std=c11 -Wall -fPIC -march=x86-64-v2 -ffast-math
    FRAMEWORKS =
endif

# Detect if compiler supports OpenMP
OPENMP_FLAG =
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    # Try clang with OpenMP first, fall back to gcc if needed
    OPENMP_TEST := $(shell $(CC) -fopenmp -E - < /dev/null 2>/dev/null && echo yes)
    ifeq ($(OPENMP_TEST),yes)
        OPENMP_FLAG = -fopenmp
    else
        # Try gcc
        GCC_EXISTS := $(shell which gcc 2>/dev/null)
        ifneq ($(GCC_EXISTS),)
            CC = gcc
            OPENMP_FLAG = -fopenmp
        endif
    endif
endif

CFLAGS = $(CFLAGS_BASE) $(OPENMP_FLAG)

MATMUL_FILES = matmul_naive.c matmul_bitnet.c matmul_simd.c matmul_swar.c matmul_lut.c

all: model_run matmul_test inference.so

model_run: model.c $(MATMUL_FILES) matmul_common.h
	$(CC) $(CFLAGS) -o model_run model.c $(MATMUL_FILES) $(FRAMEWORKS)

matmul_test: matmul_test.c $(MATMUL_FILES) matmul_common.h
	$(CC) $(CFLAGS) -o matmul_test matmul_test.c $(MATMUL_FILES) $(FRAMEWORKS)

inference.so: model_infer.c model_infer.h matmul_common.h matmul_common.c
	$(CC) $(CFLAGS) -shared -o $@ model_infer.c matmul_common.c -lm

matmul_naive.o: matmul_naive.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_naive.o matmul_naive.c

matmul_bitnet.o: matmul_bitnet.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_bitnet.o matmul_bitnet.c

matmul_simd.o: matmul_simd.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_simd.o matmul_simd.c

matmul_swar.o: matmul_swar.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_swar.o matmul_swar.c

matmul_lut.o: matmul_lut.c matmul_common.h
	$(CC) $(CFLAGS) -c -o matmul_lut.o matmul_lut.c

clean:
	rm -f model_run matmul_test inference.so *.o

.PHONY: all clean
