CC = clang
CFLAGS = -O3 -std=c11 -Wall -Wextra -fPIC -march=native -ffast-math -fopenmp
FRAMEWORKS = -framework Accelerate

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
