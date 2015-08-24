nvcc singleCM.cu -O3 -ltbb -Xcompiler -fopenmp -lgomp -arch=sm_35 -I /usr/local/cuda/samples/common/inc -o singleCM
