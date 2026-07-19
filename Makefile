# Detect OS for local dev (Mac vs Linux)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS_BASE = -O3 -std=c11 -Wall -fPIC -march=native -ffast-math
    LDFLAGS_EXTRA = -framework Accelerate
else
    # Linux / HF Spaces: cap at x86-64-v2 to avoid AVX-512 SIGILL on runtime nodes
    CFLAGS_BASE = -O3 -std=c11 -Wall -fPIC -march=x86-64-v2 -ffast-math
    LDFLAGS_EXTRA =
endif

# OpenMP detection
OPENMP_FLAG =
ifeq ($(UNAME_S),Linux)
    OPENMP_TEST := $(shell $(CC) -fopenmp -E - < /dev/null 2>/dev/null && echo yes)
    ifeq ($(OPENMP_TEST),yes)
        OPENMP_FLAG = -fopenmp
    else
        CC = gcc
        OPENMP_FLAG = -fopenmp
    endif
endif

CFLAGS = $(CFLAGS_BASE) $(OPENMP_FLAG)
LDFLAGS = -shared $(OPENMP_FLAG) -lm $(LDFLAGS_EXTRA)

# Single compilation target
inference.so: ml/bonsai.c ml/bonsai.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ ml/bonsai.c

clean:
	rm -f inference.so

.PHONY: clean
