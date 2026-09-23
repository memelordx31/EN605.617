# Makefile
# Variables below have defaults but are normally overridden by build.sh,
# which reads their real values from config.yaml, e.g.:
#   make SRC_DIR=src SRC_FILE=assignment.cu BUILD_DIR=build BIN_NAME=assignment.exe \
#        NVCC=nvcc ARCH=sm_10 FLAGS="-O3 -std=c++17"

SRC_DIR   ?= src
SRC_FILE  ?= assignment.cu
BUILD_DIR ?= build
BIN_NAME  ?= assignment.exe
NVCC      ?= nvcc
ARCH      ?= sm_75
FLAGS     ?= -O3 -std=c++17

SRC := $(SRC_DIR)/$(SRC_FILE)
BIN := $(BUILD_DIR)/$(BIN_NAME)

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(FLAGS) -arch=$(ARCH) -o $(BIN) $(SRC)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
