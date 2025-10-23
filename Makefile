CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp
INS = -Iinclude
TARGET = bin/svd

SRCS = main.c source/n2array.c

all: $(TARGET)
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(INS) -o $(TARGET) $(SRCS)
clean:
	rm -f $(TARGET)

.PHONY: all clean