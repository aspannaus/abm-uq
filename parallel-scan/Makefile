SRC_DIR = ./src
OBJ_DIR = ./obj
INC_DIR = ./include
BIN_DIR = .

CPPFLAGS = -I$(INC_DIR) -MMD -MP
CFLAGS = -Xcompiler -Wall -Xcompiler -rdynamic -lineinfo -g -std=c++17 -Xcompiler -fPIC
CUDAFLAGS = -rdc=true -gencode arch=compute_86,code=compute_86
LDFLAGS  = -Llib
LDLIBS   = -lm

NVCC = nvcc
CC = gcc
CXX = g++
OPT = -G

.PHONY: all clean

$(OBJ_DIR)/thrust_scan.o: $(SRC_DIR)/thrust_scan.cu
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(CPPFLAGS) -c $< -o $(OBJ_DIR)/thrust_scan.o

libthrustscan: $(OBJ_DIR)/thrust_scan.o
	$(NVCC) -shared $(CUDAFLAGS) $(LDFLAGS) -o libthrustscan.so $^ $(LDLIBS)

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d *.so
