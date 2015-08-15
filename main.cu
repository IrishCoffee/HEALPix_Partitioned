#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <thrust/sort.h>
#include <helper_cuda.h>
#include <sys/time.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "tbb/parallel_sort.h"
#include "printResult.h"
#include "kernel_functions.h"
#include "helper_functions.h"
#include "values.h"
#include "worker.h"
using namespace std;

int main(int argc, char* argv[])
{
	struct timeval start,end;

	double search_radius = 0.0056 * pi / 180.0;
	cudaDeviceProp deviceProp;

	printf("Number of host CPUs:\t%d\n",omp_get_num_procs());
//	checkCudaErrors(cudaGetDeviceCount(&GPU_N));
	printf("\n=============================\nCUDA-capable device count: %d\n",GPU_N);

	for(int i = 0; i < GPU_N; ++i)
	{
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp,i));
		printf("Device %d: \"%s\"\n",i,deviceProp.name);
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());
	}
	printf("===========================\n");
	
	readRefFile(argv[1],12);
	//0.2B objects
	ref_N = 200000000;
	
	gettimeofday(&start,NULL);
	worker_memory_allocation();
	gettimeofday(&end,NULL);
	printf("worker_memory_allocation %.3f s \n", diffTime(start,end) * 0.001 );

	gettimeofday(&start,NULL);
	worker_load_file(0);
	gettimeofday(&end,NULL);
	printf("worker_load_file %.3f s \n", diffTime(start,end) * 0.001 );
	
	gettimeofday(&start,NULL);
	worker_computeSI(search_radius);
	gettimeofday(&end,NULL);
	printf("worker_computeSI %.3f s \n", diffTime(start,end) * 0.001 );
	
	ref_dup_N = 0;
	gettimeofday(&start,NULL);
	worker_duplicateR();
	gettimeofday(&end,NULL);
	printf("worker_duplicateR %.3f s \n", diffTime(start,end) * 0.001 );

	gettimeofday(&start,NULL);
	tbb::parallel_sort(ref_dup_node,ref_dup_node + ref_dup_N,cmp);
	gettimeofday(&end,NULL);
	printf("worker_sort %.3f s \n", diffTime(start,end) * 0.001 );
	
	worker_memory_free();

}
