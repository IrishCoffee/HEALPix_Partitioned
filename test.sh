#make 
nvcc main.cu libcuda.a -O3 -ltbb -Xcompiler -fopenmp -lgomp -arch=sm_35 -I /usr/local/cuda/samples/common/inc -I /ghome/xjia/intel/impi/5.0.3.048/include64 -L /ghome/xjia/intel/impi/5.0.3.048/lib64 -lmpi -o main
#make clean
