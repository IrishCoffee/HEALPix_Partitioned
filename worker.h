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
	free(h_ref_dup_node);
}

void worker_load_file(int workerID)
{
	char table[20][64];
	strcpy(table[0],ref_table[workerID * 2]);
	int offset = ref_N / 2 / 20;

	omp_set_num_threads(20);
#pragma omp parallel 
	{
		int i = omp_get_thread_num() % 20;
		strcpy(table[i],ref_table[workerID * 2]);
		int len = strlen(table[i]);
		table[i][len-1] += i;
		if(readDataFile(table[i],h_ref_ra + i * offset,h_ref_dec + i * offset,offset) == -1)
			printf("load file %s error!\n",table[i]);
		//load the second directory
		strcpy(table[i],ref_table[workerID * 2 + 1]);
		len = strlen(table[i]);
		table[i][len-1] += i;
		if(readDataFile(table[i],h_ref_ra + (i + 20) * offset,h_ref_dec + (i + 20) * offset,offset) == -1)
			printf("load file %s error!\n",table[i]);
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

		double free = free_mem * 1.0 - GBSize;
		int part_N = min(part_ref_N,(int)floor(free * 1.0 / (8 + 8 + 4 * MAX_RANGE_PAIR)));
		int iteration = (int) ceil(part_ref_N * 1.0 / part_N);

		checkCudaErrors(cudaMalloc(&d_ref_ra[i],sizeof(double) * part_N));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],sizeof(double) * part_N));
		checkCudaErrors(cudaMalloc(&d_ref_range[i],sizeof(int) * part_N * MAX_RANGE_PAIR));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		for(int k = 0; k < iteration; ++k)
		{
			int size; 
			if(k != iteration - 1)
				size = part_N;
			else
				size = part_ref_N - k * part_N;
			checkCudaErrors(cudaMemcpy(d_ref_ra[i],h_ref_ra + i * part_ref_N + k * part_N, size * sizeof(double), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_dec[i],h_ref_dec + i * part_ref_N + k * part_N, size * sizeof(double), cudaMemcpyHostToDevice));

			dim3 block(512);
			dim3 grid(min(65536,size / block.x));
			get_PixRange<<<grid,block>>>(d_ref_ra[i],d_ref_dec[i],d_ref_range[i],search_radius,size);

			checkCudaErrors(cudaMemcpy(h_ref_range + i * part_ref_N * MAX_RANGE_PAIR + k * part_N * MAX_RANGE_PAIR, d_ref_range[i],size * MAX_RANGE_PAIR * sizeof(int), cudaMemcpyDeviceToHost));

		}
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
	free(h_ref_ra);
	free(h_ref_dec);
	free(h_ref_range);
	printf("ref_dup_N % d\n",ref_dup_N);
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

void worker_merge_step1(int rank)
{
	MPI_Status status;
	if(rank % 2) //odd worker send to even worker
	{
		MPI_Send(h_R_cnt, cntSize, MPI_INT, rank + 1, 3, MPI_COMM_WORLD);
		printf("node-%d send h_R_cnt to node-%d successfully\n",rank,rank +1);
	}
	else
	{
		h_R_cnt_recv = (int*)malloc(sizeof(int) * cntSize);
		MPI_Recv(h_R_cnt_recv, cntSize,MPI_INT,rank - 1,3,MPI_COMM_WORLD,&status);

		printf("node-%d recv h_R_cnt from node-%d successfully\n",rank,rank -1 );

		h_R_cnt_merged = (int*)malloc(sizeof(int) * cntSize);
		for(int i = 0; i < cntSize; ++i)
			h_R_cnt_merged[i] = h_R_cnt[i] + h_R_cnt_recv[i];

		for(int i = 0; i < 20; ++i)
			printf("%d cnt %d\n",i,h_R_cnt_merged[i]);
	}
}
#endif
