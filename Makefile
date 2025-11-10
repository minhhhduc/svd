## Simple Makefile for building main and test executables on Windows (cmd.exe)

CC := gcc
CFLAGS := -Iinclude -std=c11 -Wall -O2 -fopenmp
LDFLAGS := -lm -fopenmp
SHELL := C:/Windows/System32/cmd.exe
.SHELLFLAGS := /c

# Binary output directory
BIN_DIR := bin

# Main target: build main.c -> bin/main.exe
MAIN_SRCS := main.c source/n2array.c source/numc.c
MAIN_BIN := $(BIN_DIR)/main.exe

# Tests: each test/*.c -> _<basename>.exe (e.g. test/test_numc.c -> _test_numc.exe)
TEST_SRCS := $(wildcard test/*.c)
# Each test/NAME.c -> bin/_NAME.exe
TEST_BINS := $(patsubst test/%.c,$(BIN_DIR)/_%.exe,$(TEST_SRCS))

.PHONY: all main tests test clean

all: main

main: $(MAIN_BIN)


$(MAIN_BIN): $(MAIN_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(MAIN_SRCS) -o $@ $(LDFLAGS)

tests: $(TEST_BINS)

# Ensure bin directory exists
$(BIN_DIR):
	@if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"

# Pattern rule: compile test/NAME.c -> bin/_NAME.exe
$(BIN_DIR)/_%.exe: test/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $< source/n2array.c source/numc.c -o $@ $(LDFLAGS)

clean:
	@if exist "bin\main.exe" del /Q "bin\main.exe"
	@for /d %%d in (bin) do @if exist "%%d\_*.exe" del /Q "%%d\_*.exe"

# Print what will be built
print:
	@echo Main: $(MAIN_BIN)
	@echo Tests: $(TEST_BINS)

