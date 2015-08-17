#ifndef WORKER_H
#define WORKER_H

#include "helper_functions.h"
#include "values.h"

void worker_memory_allocation()
{
	h_ref_ra = (double*) malloc(sizeof(double) * ref_N);
	h_ref_dec = (double*) malloc(sizeof(double) * ref_N);
	h_ref_range = (int*) malloc(sizeof(int) * ref_N * MAX_RANGE_PAIR);
}

void worker_memory_free()
{
	free(h_ref_ra);
	free(h_ref_dec);
	free(h_ref_range);
	free(h_ref_dup_node);
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

		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaMalloc(&d_ref_ra[i],sizeof(double) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],sizeof(double) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_range[i],sizeof(int) * part_ref_N * MAX_RANGE_PAIR));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);


		checkCudaErrors(cudaMemcpy(d_ref_ra[i],h_ref_ra + i * part_ref_N, part_ref_N * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_ref_dec[i],h_ref_dec + i * part_ref_N, part_ref_N * sizeof(double), cudaMemcpyHostToDevice));

		dim3 block(512);
		dim3 grid(min(65536,part_ref_N / block.x));
		get_PixRange<<<grid,block>>>(d_ref_ra[i],d_ref_dec[i],d_ref_range[i],search_radius,part_ref_N);

		checkCudaErrors(cudaMemcpy(h_ref_range + i * part_ref_N * MAX_RANGE_PAIR, d_ref_range[i],part_ref_N * MAX_RANGE_PAIR * sizeof(int), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaFree(d_ref_ra[i]));
		checkCudaErrors(cudaFree(d_ref_dec[i]));
		checkCudaErrors(cudaFree(d_ref_range[i]));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);
	}
}

void worker_duplicateR()
{
	h_ref_dup_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * ref_N * 10);

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
				h_ref_dup_node[ref_dup_N].ra = h_ref_ra[i];
				h_ref_dup_node[ref_dup_N].dec = h_ref_dec[i];
				h_ref_dup_node[ref_dup_N].pix = k;
				ref_dup_N++;
			}
		}
	}

	/*
	for(int i = 0; i < 20; ++i)
		printf("%.3lf %.3lf %d\n",h_ref_dup_node[i].ra,h_ref_dup_node[i].dec,h_ref_dup_node[i].pix);
	*/
	printf("ref_dup_N % d\n",ref_dup_N);
	//parallel sort array h_ref_dup_node based on the HEALPix id
	//	tbb::parallel_sort(h_ref_dup_node,h_ref_dup_node + pos,cmp);
}

void worker_countR()
{
	h_R_cnt = (int*)malloc(sizeof(int) * cntSize);
	h_R_startPos = (int*)malloc(sizeof(int) * cntSize);
	memset(h_R_startPos,0,sizeof(int) * cntSize);
	memset(h_R_cnt,0,sizeof(int) * cntSize);

	int pix,pre_pix = -1;
	for(int i = 0; i < ref_dup_N; ++i)
	{
		pix = h_ref_dup_node[i].pix;
		h_R_cnt[pix]++;
		if(pix != pre_pix)
			h_R_startPos[pix] = i;
		pre_pix = pix;
	}
}
#endif
