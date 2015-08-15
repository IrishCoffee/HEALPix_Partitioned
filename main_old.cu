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
#include "printResult.h"
#include "kernel_functions.h"
#include "helper_functions.h"
using namespace std;

int main(int argc, char* argv[])
{
	struct timeval whole_start,whole_end;
	struct timeval cpu_start,cpu_end;
	struct timeval exec_start,exec_end;
	gettimeofday(&whole_start,NULL);

	int GBSize = 1024 * 1024 * 1024;
	int GPU_N;
	double search_radius = 0.0056 * pi / 180.0;
	cudaDeviceProp deviceProp;

	printf("Number of host CPUs:\t%d\n",omp_get_num_procs());
	checkCudaErrors(cudaGetDeviceCount(&GPU_N));
	printf("\n=============================\nCUDA-capable device count: %d\n",GPU_N);

	for(int i = 0; i < GPU_N; ++i)
	{
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp,i));
		printf("Device %d: \"%s\"\n",i,deviceProp.name);
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());
	}
	printf("===========================\n");

	omp_set_num_threads(GPU_N);

	char* refTable;
	char* samTable;
	int ref_N,sam_N;

	//host memory namespace
	int *h_ref_id,*h_ref_cnt,*h_ref_result;
	double *h_ref_ra,*h_ref_dec;
	int *h_sam_pix,*h_sam_id;
	double *h_sam_ra,*h_sam_dec;

	//device memory namespace for multiple nodes
	int *d_ref_cnt[GPU_N],*d_ref_range[GPU_N],*d_ref_result[GPU_N];
	double *d_ref_ra[GPU_N],*d_ref_dec[GPU_N];
	int *d_sam_pix[GPU_N],*d_sam_pix2[GPU_N],*d_sam_pix3[GPU_N],*d_sam_id[GPU_N];
	double *d_sam_ra[GPU_N],*d_sam_dec[GPU_N];

	int max_partN_ref,max_partN_sam;
	int partN_ref[GPU_N],partN_sam[GPU_N];

	refTable = argv[1];
	ref_N = atoi(argv[2]);
	samTable = argv[3];
	sam_N = atoi(argv[4]);

	// set number of elements that each card will process (for multi-GPU)
	max_partN_ref = ceil(ref_N * 1.0 / GPU_N);
	max_partN_sam = ceil(sam_N * 1.0 / GPU_N);
	partN_ref[GPU_N - 1] = ref_N - max_partN_ref * (GPU_N - 1);
	partN_sam[GPU_N - 1] = sam_N - max_partN_sam * (GPU_N - 1);
	for(int i = 0; i < GPU_N - 1; ++i)
	{
		partN_ref[i] = max_partN_ref;
		partN_sam[i] = max_partN_sam;
	}

	//use page-able memory
	h_ref_id = (int*)malloc(sizeof(int) * ref_N);
	h_ref_dec = (double*)malloc(sizeof(double) * ref_N);
	h_ref_ra = (double*)malloc(sizeof(double) * ref_N);
	h_ref_cnt = (int*)malloc(ref_N * sizeof(int));
	h_ref_result = (int*)malloc(ref_N * MAX_MATCH * sizeof(int));

	h_sam_pix = (int*)malloc(sam_N * sizeof(int));
	h_sam_id = (int*)malloc(sam_N * sizeof(int));
	h_sam_ra = (double*)malloc(sam_N * sizeof(double));
	h_sam_dec = (double*)malloc(sam_N * sizeof(double));

#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaMalloc(&d_ref_cnt[i],partN_ref[i] * sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_ref_range[i],partN_ref[i] * MAX_RANGE_PAIR * sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_ref_result[i],partN_ref[i] * MAX_MATCH * sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_ref_ra[i],partN_ref[i] * sizeof(double)));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],partN_ref[i] * sizeof(double)));

		checkCudaErrors(cudaMalloc(&d_sam_pix[i],max_partN_sam * sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_sam_pix2[i],max_partN_sam * sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_sam_pix3[i],max_partN_sam * sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_sam_id[i],max_partN_sam * sizeof(int)));
		checkCudaErrors(cudaMalloc(&d_sam_ra[i],max_partN_sam * sizeof(double)));
		checkCudaErrors(cudaMalloc(&d_sam_dec[i],max_partN_sam * sizeof(double)));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);
	}

	gettimeofday(&cpu_start,NULL);
	//read data from files
	readFile(refTable,h_ref_ra,h_ref_dec,h_ref_id,ref_N);
	cout << 192 << endl;
	readFile(samTable,h_sam_ra,h_sam_dec,h_sam_id,sam_N);

	gettimeofday(&cpu_end,NULL);
	printf("Read file time %.3f ms\n",diffTime(cpu_start,cpu_end));


	gettimeofday(&exec_start,NULL);
	gettimeofday(&cpu_start,NULL);
	dim3 grid(ceil(max_partN_ref * 1.0 / 512));
	dim3 block(512);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));

		checkCudaErrors(cudaMemcpy(d_ref_ra[i],h_ref_ra + i * max_partN_ref,partN_ref[i] * sizeof(double),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_ref_dec[i],h_ref_dec + i * max_partN_ref,partN_ref[i] * sizeof(double),cudaMemcpyHostToDevice));

		get_PixRange<<<grid,block>>>(d_ref_ra[i],d_ref_dec[i],d_ref_range[i],search_radius,partN_ref[i]);
		if(cudaSuccess != cudaGetLastError())
			printf("Card %d Kernel get_PixRange error!\n",i);

		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemset(d_ref_cnt[i],0,sizeof(int) * partN_ref[i]));
	}
	gettimeofday(&cpu_end,NULL);
	printf("get_PixRange time %.3f ms\n",diffTime(cpu_start,cpu_end));


	gettimeofday(&cpu_start,NULL);
	grid.x = ceil(max_partN_sam * 1.0 / 512);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));

		checkCudaErrors(cudaMemcpy(d_sam_ra[i],h_sam_ra + i * max_partN_sam, partN_sam[i] * sizeof(double),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_sam_dec[i],h_sam_dec + i * max_partN_sam, partN_sam[i] * sizeof(double),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_sam_id[i],h_sam_id + i * max_partN_sam, partN_sam[i] * sizeof(int),cudaMemcpyHostToDevice));
		getPix<<<grid,block>>>(d_sam_ra[i],d_sam_dec[i],d_sam_pix[i],partN_sam[i]);
		if(cudaSuccess != cudaGetLastError())
			printf("Card %d Kernel GetPix error!\n",i);

		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(d_sam_pix2[i],d_sam_pix[i],sizeof(int) * partN_sam[i],cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_sam_pix3[i],d_sam_pix[i],sizeof(int) * partN_sam[i],cudaMemcpyDeviceToDevice));

		thrust::device_ptr<int> dev_pix(d_sam_pix[i]);
		thrust::device_ptr<int> dev_pix2(d_sam_pix2[i]);
		thrust::device_ptr<int> dev_pix3(d_sam_pix3[i]);
		thrust::device_ptr<int> dev_id(d_sam_id[i]);
		thrust::device_ptr<double> dev_ra(d_sam_ra[i]);
		thrust::device_ptr<double> dev_dec(d_sam_dec[i]);

		thrust::sort_by_key(dev_pix,dev_pix + partN_sam[i],dev_ra);
		thrust::sort_by_key(dev_pix2,dev_pix2 + partN_sam[i],dev_dec);
		thrust::sort_by_key(dev_pix3,dev_pix3 + partN_sam[i],dev_id);

		checkCudaErrors(cudaMemcpy(h_sam_pix + i * max_partN_sam,d_sam_pix[i],partN_sam[i] * sizeof(int),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_sam_id + i * max_partN_sam,d_sam_id[i],partN_sam[i] * sizeof(int),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_sam_ra + i * max_partN_sam,d_sam_ra[i],partN_sam[i] * sizeof(double),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_sam_dec + i * max_partN_sam,d_sam_dec[i],partN_sam[i] * sizeof(double),cudaMemcpyDeviceToHost));
	}
	gettimeofday(&cpu_end,NULL);
	printf("get_Pix time %.3f ms\n",diffTime(cpu_start,cpu_end));

	//Query

	gettimeofday(&cpu_start,NULL);
	grid.x = ceil(max_partN_ref * 1.0 / 512);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));

		for(int j = 0; j < GPU_N; ++j)
		{
			checkCudaErrors(cudaMemcpy(d_sam_pix[i],h_sam_pix + j * max_partN_sam,partN_sam[j] * sizeof(int),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_sam_ra[i],h_sam_ra + j * max_partN_sam,partN_sam[j] * sizeof(double),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_sam_dec[i],h_sam_dec + j * max_partN_sam,partN_sam[j] * sizeof(double),cudaMemcpyHostToDevice));
			query_disc<<<grid,block>>>(d_ref_ra[i],d_ref_dec[i],d_sam_pix[i],d_sam_ra[i],d_sam_dec[i],d_ref_cnt[i],d_ref_range[i],d_ref_result[i],search_radius,partN_ref[i],partN_sam[j],j * max_partN_sam);
			if(cudaSuccess != cudaGetLastError())
				printf("Device %d step %d kernel query_disc error!\n",i,j);
			checkCudaErrors(cudaDeviceSynchronize());
		} 
	}
	gettimeofday(&cpu_end,NULL);
	printf("query_disc time %.3f ms\n",diffTime(cpu_start,cpu_end));

	gettimeofday(&cpu_start,NULL);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));

		checkCudaErrors(cudaMemcpy(h_ref_result + i * max_partN_ref * MAX_MATCH,d_ref_result[i], partN_ref[i] * MAX_MATCH * sizeof(int),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_ref_cnt + i * max_partN_ref,d_ref_cnt[i],partN_ref[i] * sizeof(int),cudaMemcpyDeviceToHost));
	}
	gettimeofday(&cpu_end,NULL);
	printf("Copy back result time %.3f ms\n",diffTime(cpu_start,cpu_end));

	gettimeofday(&exec_end,NULL);
	printf("total execution time %.3f ms\n",diffTime(exec_start,exec_end));

	gettimeofday(&cpu_start,NULL);
	cout << "Sum " << get_matchedCount(h_ref_cnt,ref_N) << endl;
	//	print_pix(h_sam_pix,h_sam_id,h_sam_ra,h_sam_dec,sam_N);
	gettimeofday(&cpu_end,NULL);
	printf("write file time %.3f ms\n",diffTime(cpu_start,cpu_end));

	//free page_able memory
	free(h_ref_id);
	free(h_ref_cnt);
	free(h_ref_result);
	free(h_ref_ra);
	free(h_ref_dec);
	free(h_sam_pix);
	free(h_sam_id);
	free(h_sam_ra);
	free(h_sam_dec);

#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaFree(d_ref_cnt[i]));
		checkCudaErrors(cudaFree(d_ref_range[i]));
		checkCudaErrors(cudaFree(d_ref_result[i]));
		checkCudaErrors(cudaFree(d_ref_ra[i]));
		checkCudaErrors(cudaFree(d_ref_dec[i]));
		checkCudaErrors(cudaFree(d_sam_pix[i]));
		checkCudaErrors(cudaFree(d_sam_pix2[i]));
		checkCudaErrors(cudaFree(d_sam_pix3[i]));
		checkCudaErrors(cudaFree(d_sam_id[i]));

		checkCudaErrors(cudaDeviceReset());
	}
	gettimeofday(&whole_end,NULL);
	printf("whole time %.3f ms\n",diffTime(whole_start,whole_end));
	return 0;
}
