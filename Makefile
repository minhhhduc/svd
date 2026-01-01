## Simple Makefile for building main and test executables on Windows (cmd.exe)

CC := gcc
CFLAGS := -Iinclude -std=c11 -Wall -O2 -fopenmp
LDFLAGS := -lm -fopenmp
SHELL := C:/Windows/System32/cmd.exe
.SHELLFLAGS := /c

# Binary output directory
BIN_DIR := bin
SOURCE_DIR := source

# Main target: build main.c -> bin/main.exe
MAIN_SRCS := main.c source/n2array.c source/numc.c
MAIN_BIN := $(BIN_DIR)/main.exe

# Parallel object files to compile
PARALLEL_OBJS := source/SVD_parallel.o source/other_operating_parallel.o source/norm_reducing_jacobi_parallel.o

# Tests: each test/*.c -> _<basename>.exe (e.g. test/test_numc.c -> _test_numc.exe)
TEST_SRCS := $(wildcard test/*.c)
# Each test/NAME.c -> bin/_NAME.exe
TEST_BINS := $(patsubst test/%.c,$(BIN_DIR)/_%.exe,$(TEST_SRCS))

.PHONY: all main tests test clean build-parallel

all: main

main: $(MAIN_BIN)

# Build parallel object files
build-parallel: $(PARALLEL_OBJS)
	@echo Parallel object files built: $(PARALLEL_OBJS)

# Pattern rule for compiling .c to .o
source/%.o: source/%.c | $(SOURCE_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN_BIN): $(MAIN_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(MAIN_SRCS) -o $@ $(LDFLAGS)

tests: $(TEST_BINS)

# Ensure bin directory exists
$(BIN_DIR):
	@if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"

# Ensure source directory exists
$(SOURCE_DIR):
	@if not exist "$(SOURCE_DIR)" mkdir "$(SOURCE_DIR)"

# Pattern rule: compile test/NAME.c -> bin/_NAME.exe
# Skip files without main() by ignoring compilation errors with -
$(BIN_DIR)/_%.exe: test/%.c | $(BIN_DIR)
	-$(CC) $(CFLAGS) $< source/n2array.c source/numc.c -o $@ $(LDFLAGS) 2>nul

clean:
	@if exist "bin\main.exe" del /Q "bin\main.exe"
	@for /d %%d in (bin) do @if exist "%%d\_*.exe" del /Q "%%d\_*.exe"
	@if exist "source\SVD_parallel.o" del /Q "source\SVD_parallel.o"
	@if exist "source\other_operating_parallel.o" del /Q "source\other_operating_parallel.o"
	@if exist "source\norm_reducing_jacobi_parallel.o" del /Q "source\norm_reducing_jacobi_parallel.o"

# Print what will be built
print:
	@echo Main: $(MAIN_BIN)
	@echo Tests: $(TEST_BINS)
	@echo Parallel Objects: $(PARALLEL_OBJS)

