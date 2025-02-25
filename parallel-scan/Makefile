SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .

EXE = $(BIN_DIR)/scan.x
SRC = $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

CPPFLAGS := -Iinclude -MMD -MP
CFLAGS = -Xcompiler -Wall -Xcompiler -rdynamic -lineinfo -g -std=c++17 -Xcompiler -fPIC
CUDAFLAGS = -rdc=true -gencode arch=compute_86,code=compute_86
LDFLAGS  = -Llib
LDLIBS   = -lm

NVCC = nvcc
CC = gcc
CXX = g++
OPT = -G

.PHONY: all clean

all: $(EXE)

objs: $(OBJS) $(SRC)
	echo $<
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(CPPFLAGS) -c $^ -o $(OBJ_DIR)/$@

kernels.o: $(SRC_DIR)/kernels.cu
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(CPPFLAGS) -c $< -o $(OBJ_DIR)/kernels.o

scan.o: $(SRC_DIR)/scan.cu
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(CPPFLAGS) -c $< -o $(OBJ_DIR)/scan.o

utils.o: $(SRC_DIR)/utils.cu 
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) $(CPPFLAGS) -c $< -o $(OBJ_DIR)/utils.o

libscan: $(OBJ)
	$(NVCC) -shared $(CUDAFLAGS) $(LDFLAGS) -o libscan.so $^ $(LDLIBS)

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d
