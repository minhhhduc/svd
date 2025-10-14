# Makefile for parallel_computing/final_project
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fopenmp -Iinclude
LDFLAGS := -fopenmp

# Prevent recursive make from printing "Entering directory" / "Leaving directory"
MAKEFLAGS += --no-print-directory

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

SOURCES_CPP := main.cpp source/n2array.cpp source/numc.cpp
OBJECTS := $(SOURCES_CPP:.cpp=.o)
TEST_OBJECTS := $(filter-out main.o,$(OBJECTS))

# Build main executable (compile sources)
$(BIN_DIR)/main: $(OBJECTS) | $(BIN_DIR)
	@$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

# Build all test executables: compile every .cpp in test/ into bin/<basename>
TEST_SOURCES := $(wildcard test/*.cpp)
TEST_BASENAMES := $(notdir $(TEST_SOURCES:.cpp=))
# Prefix test binaries with t_ so they can be discovered easily (bin/t_<name>)
TEST_BINS := $(addprefix $(BIN_DIR)/t_,$(TEST_BASENAMES))

test:
	@for %%f in (test\*.cpp) do @( \
		( findstr /C:"int main" "%%f" >nul 2>&1 || findstr /C:"void main" "%%f" >nul 2>&1 ) \
		&& ( $(MAKE) $(BIN_DIR)/t_%%~nf ) \
	)

$(BIN_DIR)/t_%: test/%.cpp $(TEST_OBJECTS) | $(BIN_DIR)
	@$(CXX) $(CXXFLAGS) test/$*.cpp $(TEST_OBJECTS) -o $@ $(LDFLAGS)

# Optional: compile source files to objects (for non-header-only parts)
%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@if exist $(BIN_DIR) rmdir /s /q $(BIN_DIR)
	@if exist library rmdir /s /q library