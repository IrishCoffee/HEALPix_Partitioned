################################################################################
#
# Copyright 1993-3512 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7352-1 through 
# 227.7352-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OSX and Linux Platforms
#
################################################################################

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-6.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++
# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=

# CUDA code generation flags
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   :=  $(GENCODE_SM35) 
#$(GENCODE_SM35)

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudadevrt
else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudadevrt
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      CCFLAGS   += -m32
      NVCCFLAGS := -m32 -dc
else
      CCFLAGS   += -m64
      NVCCFLAGS := -m64 -dc
endif

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      TARGET    := release
endif

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I. -I.. -I$(CUDA_PATH)/samples/common/inc

noinst_LIBRARIES = libcuda.a 

noinst_OBJECTS = kernel_functions.o healpix_base.o geometry_cal.o vec3.o singleCM_kernel.o

libcuda_a_SOURCES = kernel_functions.cu healpix_base.cu geometry_cal.cu vec3.cu singleCM_kernel.cu

# Target rules
all: build

build: noinst_LIBRARIES

noinst_LIBRARIES: $(noinst_OBJECTS)
	ar cru $(noinst_LIBRARIES) $(noinst_OBJECTS)

kernel_functions.o: kernel_functions.cu
	$(NVCC) -dc -O3 kernel_functions.cu $(GENCODE_SM35)

healpix_base.o: healpix_base.cu 
	$(NVCC) -dc -O3 healpix_base.cu $(GENCODE_SM35)

geometry_cal.o: geometry_cal.cu
	$(NVCC) -dc -O3 geometry_cal.cu $(GENCODE_SM35)

vec3.o: vec3.cu
	$(NVCC) -dc -O3 vec3.cu $(GENCODE_SM35)

singleCM_kernel.o: singleCM_kernel.cu
	$(NVCC) -dc -O3 singleCM_kernel.cu $(GENCODE_SM35)

clean:
	rm -f $(noinst_LIBRARIES) *.o 

install:
  
