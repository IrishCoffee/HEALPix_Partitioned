#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#include "tbb/parallel_sort.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
using namespace std;

int ref_line[20];
char ref_file[20][16];
int sam_line[20];
char sam_file[20][16];
const int GPU_N = 2;
const int GBSize = 1024 * 1024 * 1024;
struct NODE
{
	double ra,dec;
	int pix;
};

bool cmp(NODE a,NODE b)
{
	return a.pix < b.pix;
}

void readFile(char *file,int N, NODE nn[])
{
	FILE *fd = fopen(file,"r");
	if(fd == NULL)
		printf("Read %s error!\n",file);
	for(int i = 0; i < N; ++i)
		fscanf(fd,"%d%lf%lf",&nn[i].pix,&nn[i].ra,&nn[i].dec);
	fclose(fd);
}
	__host__ __device__
int begin_index(int key, NODE *node, int N)
{
	for(int i = 0; i < N; ++i)
		if(node[i].pix > key)
			return i;
	return N;
}

	__host__ __device__
int binary_search(int key, NODE *node, int N)
{
	int st = 0;
	int ed = N - 1;
	while(st < ed)
	{
		int mid = st + ((ed - st) >> 1);
		if(node[mid].pix <= key)
			st = mid + 1;
		else
			ed = mid;
	}
	if(node[ed].pix > key)
		return ed;
	return -1;
}
	__global__
void kernel_singleCM(NODE *ref_node, int ref_N, NODE *sam_node, int sam_N, int *sam_match,int *sam_matchedCnt)
{
	__shared__ int sam_pix[1024];
	__shared__ double sam_ra[1024];
	__shared__ double sam_dec[1024];
	__shared__ int sam_cnt[1024];

	__shared__ int start_pix,end_pix;
	__shared__ int start_ref_pos,end_ref_pos;
	__shared__ int block_sam_N,block_ref_N;
	__shared__ int tmp_start,tmp_end;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < sam_N)
	{
		sam_cnt[threadIdx.x] = 0;
		sam_pix[threadIdx.x] = sam_node[tid].pix;
		sam_ra[threadIdx.x] = sam_node[tid].ra;
		sam_dec[threadIdx.x] = sam_node[tid].dec;
	}
	if(threadIdx.x == 0)
	{
		if(blockIdx.x == gridDim.x - 1)
			block_sam_N = sam_N - blockIdx.x * blockDim.x;
		else
			block_sam_N = blockDim.x;

		start_pix = sam_pix[0];
		end_pix = sam_pix[block_sam_N - 1];

		if(start_pix == 0)
			start_ref_pos = 0;
		else
			start_ref_pos = binary_search(start_pix - 1,ref_node,ref_N);

		tmp_start = begin_index(start_pix - 1,ref_node,ref_N);
		if(start_pix == 0)
			tmp_start = 0;
		tmp_end = begin_index(end_pix,ref_node,ref_N) - 1;

		end_ref_pos = binary_search(end_pix,ref_node,ref_N) - 1;
		if(end_ref_pos == -2)
			end_ref_pos = ref_N - 1;

		block_ref_N = end_ref_pos - start_ref_pos + 1;
	}
	syncthreads();
	if(threadIdx.x == 0  && blockIdx.x < 5)
		//	printf("\n\nstart_sam_pix %d end_sam_pix %d \nstart_ref_pix %d end_ref_pix %d\ntmp_start_pix %d tmp_end_pix %d\nstart_ref_pos %d end_ref_pos %d\ntmp_start %d tmp_end %d\nblock_sam_N %d block_ref_N %d\n",start_pix,end_pix,ref_node[start_ref_pos].pix,ref_node[end_ref_pos].pix,ref_node[tmp_start].pix,ref_node[tmp_end].pix,start_ref_pos,end_ref_pos,tmp_start,tmp_end,block_sam_N,block_ref_N);
	{
		printf("\n\nstart_sam_pix %d end_sam_pix %d \nstart_ref_pix %d end_ref_pix %d\ntmp_start_pix %d tmp_end_pix %d\n",start_pix,end_pix,ref_node[start_ref_pos].pix,ref_node[end_ref_pos].pix,ref_node[tmp_start].pix,ref_node[tmp_end].pix);
		printf("start_ref_pos %d end_ref_pos %d\ntmp_start %d tmp_end %d\nblock_sam_N %d block_ref_N %d\n",start_ref_pos,end_ref_pos,tmp_start,tmp_end,block_sam_N,block_ref_N);
	}
}

void singleCM(NODE h_ref_node[], int ref_N, NODE h_sam_node[], int sam_N, int h_sam_match[],int h_sam_matchedCnt[])
{
	//the maximum number of sample points that can be matched each time by each card
	int part_sam_N = 50000000;
	int part_ref_N = 8 * part_sam_N;

	NODE *d_ref_node[2];
	NODE *d_sam_node[2];
	int *d_sam_match[2], *d_sam_matchedCnt[2];

	/*
	   omp_set_num_threads(2);
#pragma omp parallel
{
int i = omp_get_thread_num() % GPU_N;
	 */
	for(int i = 0; i < GPU_N; ++i)
	{
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);


		checkCudaErrors(cudaMalloc(&d_ref_node[i],sizeof(NODE) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_sam_node[i],sizeof(NODE) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_match[i],sizeof(int) * part_sam_N  * 5));
		checkCudaErrors(cudaMalloc(&d_sam_matchedCnt[i],sizeof(int) * part_sam_N));
		checkCudaErrors(cudaMemset(d_sam_matchedCnt[i],0,sizeof(int) * part_sam_N));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d after malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		//the total number of sample points processed by this card
		int card_sam_N;
		if(i == GPU_N - 1)
			card_sam_N = sam_N - i * sam_N / GPU_N;
		else
			card_sam_N = sam_N / GPU_N;

		int iteration = ceil(card_sam_N * 1.0 / part_sam_N);

		for(int ite = 0; ite < iteration; ++ite)
		{
			int cur_sam_N;
			if(ite == iteration - 1) // the last round
				cur_sam_N = card_sam_N - ite * part_sam_N;
			else
				cur_sam_N = part_sam_N;

			int start_sam_pos = ite * part_sam_N + i * sam_N / GPU_N;
			int end_sam_pos = start_sam_pos + cur_sam_N - 1;

			int start_pix = h_sam_node[start_sam_pos].pix;
			int end_pix = h_sam_node[end_sam_pos].pix;

			int start_ref_pos;
			if(start_pix == 0)
				start_ref_pos = 0;
			else
				//		start_ref_pos = begin_index(start_pix - 1,h_ref_node,ref_N);
				start_ref_pos = binary_search(start_pix - 1,h_ref_node,ref_N);

			//no reference point need to be matched
			if(start_ref_pos == -1)
				break;
			//	int end_ref_pos = begin_index(end_pix,h_ref_node,ref_N) - 1;
			int end_ref_pos = binary_search(end_pix,h_ref_node,ref_N) - 1;
			if(end_ref_pos == -2)
				end_ref_pos = ref_N - 1;
			int cur_ref_N = end_ref_pos - start_ref_pos + 1;

			dim3 block(1024);
			dim3 grid(min(65536,(int)ceil(cur_sam_N * 1.0 / block.x)));

			printf("\n\nCard %d iteration %d\n",i,ite);
			printf("block.x %d grid.x %d\n",block.x,grid.x);
			printf("start_sam_pos %d start_sam_pix %d end_sam_pos %d end_sam_pix %d sam_N %d\n",start_sam_pos,start_pix,end_sam_pos,end_pix,cur_sam_N);
			printf("start_ref_pos %d start_ref_pix %d end_ref_pos %d end_ref_pix %d ref_N %d\n",start_ref_pos,h_ref_node[start_ref_pos].pix,end_ref_pos,h_ref_node[end_ref_pos].pix,cur_ref_N);

			checkCudaErrors(cudaMemcpy(d_sam_node[i],h_sam_node + start_sam_pos,cur_sam_N * sizeof(NODE),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_node[i],h_ref_node + start_ref_pos,cur_ref_N * sizeof(NODE), cudaMemcpyHostToDevice));
			kernel_singleCM<<<grid,block>>>(d_ref_node[i],cur_ref_N,d_sam_node[i],cur_sam_N,d_sam_match[i],d_sam_matchedCnt[i]);
		}
	}
}




int main(int argc, char *argv[])
{

	const int ref_N = 1538557732;
	const int sam_N = 200003876;
	time_t rawtime;

	FILE *fd = fopen(argv[1],"r");
	for(int i = 0; i < 20; ++i)
		fscanf(fd,"%d%s",&ref_line[i],ref_file[i]);
	fclose(fd);

	fd = fopen(argv[2],"r");
	for(int i = 0; i < 20; ++i)
		fscanf(fd,"%d%s",&sam_line[i],sam_file[i]);
	fclose(fd);

	NODE *ref_node,*sam_node;
	int *sam_matchedCnt;
	int *sam_match;

	ref_node = (NODE *)malloc(sizeof(NODE) * ref_N);
	sam_node = (NODE *)malloc(sizeof(NODE) * sam_N);

	sam_matchedCnt = (int *)malloc(sizeof(int) * sam_N);
	sam_match = (int *)malloc(sizeof(int) * sam_N * 5);


	time(&rawtime);
	printf("before read ref file : %s\n",ctime(&rawtime));

	omp_set_num_threads(20);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % 20;
		int offset = i * ref_line[0];
		readFile(ref_file[i],ref_line[i],ref_node + offset);
	}

	time(&rawtime);
	printf("after read ref file : %s\n",ctime(&rawtime));

#pragma omp parallel
	{
		int i = omp_get_thread_num() % 20;
		int offset = i * sam_line[0];
		readFile(sam_file[i],sam_line[i],sam_node + offset);
	}

	time(&rawtime);
	printf("after read sam file : %s\n",ctime(&rawtime));

	tbb::parallel_sort(ref_node,ref_node + ref_N,cmp);

	time(&rawtime);
	printf("after sort : %s\n",ctime(&rawtime));

	singleCM(ref_node,ref_N,sam_node,sam_N,sam_match,sam_matchedCnt);
	time(&rawtime);
	printf("singleCM : %s\n",ctime(&rawtime));
}
