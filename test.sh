#make 
nvcc main.cu libcuda.a -O3 -Xcompiler -fopenmp -lgomp -ltbb -arch=sm_35 -I /usr/local/cuda/samples/common/inc -o main
#make clean
