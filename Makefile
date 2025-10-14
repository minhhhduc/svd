# Makefile for parallel_computing/final_project
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fopenmp -Iinclude
LDFLAGS := -fopenmp

BIN_DIR := bin
INCLUDE_DIR := include
SOURCES := $(wildcard source/*.cpp)
OBJECTS := $(SOURCES:source/%.cpp=library/%.o)

.PHONY: all clean prepare test

all: prepare $(BIN_DIR)/main

# Create directories if they don't exist (Windows compatible)
prepare: | $(BIN_DIR) library

$(BIN_DIR):
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)

library:
	@if not exist library mkdir library

# Build main executable (header-only templates, no need for object files)
$(BIN_DIR)/main: main.cpp $(INCLUDE_DIR)/n2array.hpp $(INCLUDE_DIR)/n2array.tpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS)

# Build test executable if test files exist
test: $(BIN_DIR)/test_n2array

$(BIN_DIR)/test_n2array: test/n2array.cpp $(INCLUDE_DIR)/n2array.hpp $(INCLUDE_DIR)/n2array.tpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) test/n2array.cpp -o $@ $(LDFLAGS)

# Optional: compile source files to objects (for non-header-only parts)
library/%.o: source/%.cpp $(INCLUDE_DIR)/%.hpp | library
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@if exist $(BIN_DIR) rmdir /s /q $(BIN_DIR)
	@if exist library rmdir /s /q library