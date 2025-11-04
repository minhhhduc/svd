CC = gcc
# set USE_OPENCL=1 to compile with OpenCL support
USE_OPENCL ?= 1

CFLAGS = -Wall -Wextra -O2 -fopenmp
INS = -Iinclude
LIB = -Llib
TARGET = bin\main
TESTDIR = test

APP_SRCS = main.c
LIB_SRCS = source/n2array.c source/numc.c
ifeq ($(USE_OPENCL),1)
	CFLAGS += -DUSE_OPENCL -Iinclude/CL
	OPENCL_LIBS = -lOpenCL
	LIB_SRCS += source/opencl_helper.c
endif

SRCS = $(APP_SRCS) $(LIB_SRCS)
# Conditionally include OpenCL test file
ifeq ($(USE_OPENCL),1)
	TEST_SRCS := $(filter-out $(TESTDIR)/test.c,$(wildcard $(TESTDIR)/*.c))
else
	TEST_SRCS := $(filter-out $(TESTDIR)/test.c $(TESTDIR)/test_opencl.c,$(wildcard $(TESTDIR)/*.c))
endif

# Default: build the main binary only
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(INS) -o $(TARGET) $(SRCS) $(LIB) $(OPENCL_LIBS)

# Build the test binary (but don't run it)
$(TESTDIR)/test: $(TEST_SRCS)
	$(CC) $(CFLAGS) $(INS) -o $(TESTDIR)/test $(TEST_SRCS) $(LIB) $(LIB_SRCS) $(OPENCL_LIBS)

# Run tests (builds test binary first if needed)
test: $(TESTDIR)/test
	@echo Running tests... && "$(TESTDIR)\\test.exe"

# Convenience target: build everything and run tests
build-all-test: $(TARGET) $(TESTDIR)/test
	@echo Build complete. Running tests... && "$(TESTDIR)\\test.exe"

.PHONY: opencl-test
opencl-test:
	@echo Building and running OpenCL multiply test...
	$(CC) $(CFLAGS) $(INS) -o $(TESTDIR)/opencl_test.exe $(TESTDIR)/test_opencl.c $(LIB) $(LIB_SRCS) $(OPENCL_LIBS)
	@"$(TESTDIR)\\opencl_test.exe"

clean:
	-@if exist "$(TARGET).exe" del "$(TARGET).exe" 2>nul
	-@if exist "$(TESTDIR)\test.exe" del "$(TESTDIR)\test.exe" 2>nul

.PHONY: all clean test build-all-test