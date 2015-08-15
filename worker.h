#ifndef WORKER_H
#define WORKER_H

#include "helper_functions.h"
#include "values.h"


bool cmp(REF_NODE node_a,REF_NODE node_b)
{
	return node_a.pix < node_b.pix;
}
void worker_memory_allocation()
{
	h_ref_ra = (double*) malloc(sizeof(double) * ref_N);
	h_ref_dec = (double*) malloc(sizeof(double) * ref_N);
	h_ref_range = (int*) malloc(sizeof(int) * ref_N * MAX_RANGE_PAIR);

	//for ref_N, we only consider the even number situation by now
	int d_ref_N = ref_N / GPU_N;

	omp_set_num_threads(GPU_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaMalloc(&d_ref_ra[i],sizeof(double) * d_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],sizeof(double) * d_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_range[i],sizeof(int) * d_ref_N * MAX_RANGE_PAIR));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);
	}
}

void worker_memory_free()
{
	free(h_ref_ra);
	free(h_ref_dec);
	free(h_ref_range);
	free(ref_dup_node);

	omp_set_num_threads(GPU_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaFree(d_ref_ra[i]));
		checkCudaErrors(cudaFree(d_ref_dec[i]));
		checkCudaErrors(cudaFree(d_ref_range[i]));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);
	}
}

void worker_load_file(int workerID)
{
	char table[20][64];
	strcpy(table[0],ref_table[workerID * 2]);
	int len = strlen(table[0]);
	int offset = ref_N / 2 / 20;

	omp_set_num_threads(20);
#pragma omp parallel 
	{
		int i = omp_get_thread_num() % 20;
		strcpy(table[i],ref_table[workerID * 2]);
		table[i][len-1] += i;
		readDataFile(table[i],h_ref_ra + i * offset,h_ref_dec + i * offset,offset);

		//load the second directory
		strcpy(table[i],ref_table[workerID * 2 + 1]);
		table[i][len-1] += i;
		readDataFile(table[i],h_ref_ra + (i + 20) * offset,h_ref_dec + (i + 20) * offset,offset);
	}
	return;
}

void worker_computeSI(double search_radius)
{
	omp_set_num_threads(GPU_N);
	int part_ref_N = ref_N / GPU_N;
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));

		checkCudaErrors(cudaMemcpy(d_ref_ra[i],h_ref_ra + i * part_ref_N, part_ref_N * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_ref_dec[i],h_ref_dec + i * part_ref_N, part_ref_N * sizeof(double), cudaMemcpyHostToDevice));

		dim3 block(512);
		dim3 grid(min(65536,part_ref_N / block.x));
		get_PixRange<<<grid,block>>>(d_ref_ra[i],d_ref_dec[i],d_ref_range[i],search_radius,part_ref_N);

		checkCudaErrors(cudaMemcpy(h_ref_range + i * part_ref_N * MAX_RANGE_PAIR, d_ref_range[i],part_ref_N * MAX_RANGE_PAIR * sizeof(int), cudaMemcpyDeviceToHost));

	}
}

void worker_duplicateR()
{
	ref_dup_node = (REF_NODE *)malloc(sizeof(REF_NODE) * ref_N * 10);

	printf("before duplicate ref_dup_N %d\n",ref_dup_N);
	//#pragma omp parallel for
	for(int i = 0; i < ref_N; ++i)
	{
		int off = i * MAX_RANGE_PAIR;
		for(int j = off;j < (i+1) * MAX_RANGE_PAIR; j += 2)
		{
			if(h_ref_range[j] == -1)
				break;
			for(int k = h_ref_range[j]; k < h_ref_range[j+1]; ++k)
			{
				ref_dup_node[ref_dup_N].ra = h_ref_ra[i];
				ref_dup_node[ref_dup_N].dec = h_ref_dec[i];
				ref_dup_node[ref_dup_N].pix = k;
				ref_dup_N++;
			}
		}
	}

	for(int i = 0; i < 20; ++i)
		printf("%.3lf %.3lf %d\n",ref_dup_node[i].ra,ref_dup_node[i].dec,ref_dup_node[i].pix);
	printf("ref_dup_N % d\n",ref_dup_N);
	//parallel sort array ref_dup_node based on the HEALPix id
	//	tbb::parallel_sort(ref_dup_node,ref_dup_node + pos,cmp);
}
#endif
