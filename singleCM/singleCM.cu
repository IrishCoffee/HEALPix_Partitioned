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
#include <cstring>
#include "tbb/parallel_sort.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
using namespace std;
const double pi=3.141592653589793238462643383279502884197;
int ref_line[20];
char ref_file[20][16];
int sam_line[20];
char sam_file[20][16];
const int GPU_N = 2;
const int GBSize = 1024 * 1024 * 1024;
const int block_size = 512;
const int TILE_SIZE = 1024;
struct NODE
{
	double ra,dec;
	int pix;
};
const int cntSize = 805306368;
double diffTime(timeval start,timeval end)
{
	return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}
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
int get_start(int key,NODE *node,int N)
{
	for(int i = 0; i < N; ++i)
	{
		if(node[i].pix >= key)
			return i;
	}
	return N;
}
__host__ __device__
int get_end(int key,NODE *node,int N)
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
__host__ __device__ double radians(double degree)
{
	return degree * pi / 180.0;
}
	__host__	__device__
bool matched(double ra1,double dec1,double ra2,double dec2,double radius)
{
	double z1 = sin(radians(dec1));
	double x1 = cos(radians(dec1)) * cos(radians(ra1));
	double y1 = cos(radians(dec1)) * sin(radians(ra1));

	double z2 = sin(radians(dec2));
	double x2 = cos(radians(dec2)) * cos(radians(ra2));
	double y2 = cos(radians(dec2)) * sin(radians(ra2));

	double distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	double dist2 = 4 * pow(sin(radians(0.0056 / 2)),2);

	if(distance <= dist2)
		return true;
	return false;
}
	__global__
void kernel_singleCM(NODE *ref_node, int ref_N, NODE *sam_node, int sam_N, int *sam_match,int *sam_matchedCnt,int ref_offset,int sam_offset)
{
	__shared__ int s_ref_pix[TILE_SIZE];
	__shared__ double s_ref_ra[TILE_SIZE];
	__shared__ double s_ref_dec[TILE_SIZE];

	__shared__ int start_pix,end_pix;
	__shared__ int start_ref_pos,end_ref_pos;
	__shared__ int block_sam_N,block_ref_N;
	__shared__ int iteration;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < sam_N)
		sam_matchedCnt[tid] = 0;


	if(threadIdx.x == 0)
	{
		if(blockIdx.x == gridDim.x - 1) // the last block
			block_sam_N = sam_N - blockIdx.x * blockDim.x;
		else
			block_sam_N = blockDim.x;

		start_pix = sam_node[tid].pix;
		end_pix = sam_node[tid + block_sam_N - 1].pix;

		if(start_pix == 0)
			start_ref_pos = 0;
		else
			start_ref_pos = binary_search(start_pix - 1,ref_node,ref_N);

		end_ref_pos = binary_search(end_pix,ref_node,ref_N);

		if(end_ref_pos == -1)
			end_ref_pos = ref_N - 1;
		else
			end_ref_pos--;
		block_ref_N = end_ref_pos - start_ref_pos + 1;
		iteration = ceil(block_ref_N * 1.0 / TILE_SIZE);
	}
	__syncthreads();
	if(start_ref_pos == -1 || end_ref_pos < start_ref_pos)
		return;

	int pix,cnt = 0;
	double sam_ra,sam_dec;
	if(tid < sam_N)
	{
		pix = sam_node[tid].pix;
		sam_ra = sam_node[tid].ra;
		sam_dec = sam_node[tid].dec;
		cnt = 0;
	}
	__syncthreads();
	for(int ite = 0; ite < iteration; ++ite)
	{
		__syncthreads();
		for(int k = 0; k < TILE_SIZE / blockDim.x; ++k)
		{
			int ref_pos = start_ref_pos + ite * TILE_SIZE + blockDim.x * k + threadIdx.x;
			int s_ref_pos = blockDim.x * k + threadIdx.x;
			if(ref_pos <= end_ref_pos)
			{
				s_ref_pix[s_ref_pos] = ref_node[ref_pos].pix;
				s_ref_ra[s_ref_pos] = ref_node[ref_pos].ra;
				s_ref_dec[s_ref_pos] = ref_node[ref_pos].dec;
			}
			else
				s_ref_pix[s_ref_pos] = -1;
		}
		__syncthreads();
		if(tid >= sam_N)
			continue;
		for(int j = 0; j < TILE_SIZE; ++j)
		{
			if(s_ref_pix[j] == -1 || s_ref_pix[j] > pix)
				break;
			if(s_ref_pix[j] < pix)
				continue;
			if(s_ref_pix[j] == pix && matched(sam_ra,sam_dec,s_ref_ra[j],s_ref_dec[j],0.0056))
			{
				cnt++;
				if(cnt <= 5)
					sam_match[tid * 5 + cnt - 1] = ref_offset + start_ref_pos + ite * TILE_SIZE + j;
			}
		}
	}

	if(tid < sam_N)
		sam_matchedCnt[tid] = cnt;
}


void singleCM(NODE h_ref_node[], int ref_N, NODE h_sam_node[], int sam_N, int h_sam_match[],int h_sam_matchedCnt[])
{
	//the maximum number of sample points that can be matched each time by each card
	int part_sam_N = 25000000;
	int part_ref_N = 8 * part_sam_N;

	NODE *d_ref_node[GPU_N];
	NODE *d_sam_node[GPU_N];
	int *d_sam_match[GPU_N], *d_sam_matchedCnt[GPU_N];
	omp_set_num_threads(GPU_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);


		checkCudaErrors(cudaMalloc(&d_ref_node[i],sizeof(NODE) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_sam_node[i],sizeof(NODE) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_match[i],sizeof(int) * part_sam_N  * 5));
		checkCudaErrors(cudaMalloc(&d_sam_matchedCnt[i],sizeof(int) * part_sam_N));

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
				start_ref_pos = binary_search(start_pix - 1,h_ref_node,ref_N);
			//				start_ref_pos = get_start(start_pix,h_ref_node,ref_N);

			if(start_ref_pos == -1)
				continue;
			int end_ref_pos = binary_search(end_pix,h_ref_node,ref_N) - 1;
			if(end_ref_pos == -2)
				end_ref_pos = ref_N - 1;
			int cur_ref_N = end_ref_pos - start_ref_pos + 1;

			dim3 block(block_size);
			dim3 grid(min(65536,(int)ceil(cur_sam_N * 1.0 / block.x)));

			if(cur_ref_N == 0)
				continue;

			printf("\n\nCard %d iteration %d\n",i,ite);
			printf("block.x %d grid.x %d\n",block.x,grid.x);
			printf("start_sam_pos %d start_sam_pix %d end_sam_pos %d end_sam_pix %d sam_N %d\n",start_sam_pos,start_pix,end_sam_pos,end_pix,cur_sam_N);
			printf("start_ref_pos %d start_ref_pix %d end_ref_pos %d end_ref_pix %d ref_N %d\n",start_ref_pos,h_ref_node[start_ref_pos].pix,end_ref_pos,h_ref_node[end_ref_pos].pix,cur_ref_N);
			checkCudaErrors(cudaMemcpy(d_sam_node[i],h_sam_node + start_sam_pos,cur_sam_N * sizeof(NODE),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_node[i],h_ref_node + start_ref_pos,cur_ref_N * sizeof(NODE), cudaMemcpyHostToDevice));
			kernel_singleCM<<<grid,block>>>(d_ref_node[i],cur_ref_N,d_sam_node[i],cur_sam_N,d_sam_match[i],d_sam_matchedCnt[i],start_ref_pos,start_sam_pos);
			checkCudaErrors(cudaMemcpy(h_sam_matchedCnt + start_sam_pos,d_sam_matchedCnt[i],cur_sam_N * sizeof(int),cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(h_sam_match + start_sam_pos * 5,d_sam_match[i],cur_sam_N * 5 * sizeof(int),cudaMemcpyDeviceToHost));
		}
	}
	unsigned long long sum = 0;
	int cnt[1000];
	memset(cnt,0,sizeof(cnt));
	for(int i = sam_N - 1; i >= 0; --i)
	{
		sum += h_sam_matchedCnt[i];
		/*
		   cout << i << " " << h_sam_matchedCnt[i] << endl;
		   cout << h_sam_node[i].ra << " " << h_sam_node[i].dec << endl;
		   cout << "\n----------------\n" << endl;
		   for(int j = i * 5; j < i * 5 + min(5,h_sam_matchedCnt[i]); ++j)
		   {
		   int pos = h_sam_match[j];
		   cout << h_ref_node[pos].ra << " " << h_ref_node[pos].dec << endl;
		   }
		   cout << "\n--------------------\n" << endl;
		 */
	}
	cout << "sum " << sum << endl;
	cout << "ave " << sum * 1.0 / sam_N << endl;
}


int main(int argc, char *argv[])
{
	struct timeval start,end;

	int ref_N = atoi(argv[3]);
	int sam_N = atoi(argv[4]);
	//	const int ref_N = 200006373;
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

	gettimeofday(&start,NULL);
	tbb::parallel_sort(ref_node,ref_node + ref_N,cmp);
	tbb::parallel_sort(sam_node,sam_node + sam_N,cmp);
	gettimeofday(&end,NULL);
	printf("sort time %.3f \n",diffTime(start,end) * 0.001);

	time(&rawtime);
	printf("after sort : %s\n",ctime(&rawtime));
	
	gettimeofday(&start,NULL);
	singleCM(ref_node,ref_N,sam_node,sam_N,sam_match,sam_matchedCnt);
	gettimeofday(&end,NULL);

	printf("single CM %.3f s\n",diffTime(start,end) * 0.001);
	printf("single CM %.3f min\n",diffTime(start,end) * 0.001 / 60);
	time(&rawtime);
	printf("singleCM : %s\n",ctime(&rawtime));

	
	int *ref_match = (int*)malloc(sizeof(int) * ref_N * 5);
	int *ref_matchedCnt = (int*)malloc(sizeof(int) * ref_N);

	gettimeofday(&start,NULL);
	singleCM(sam_node,sam_N,ref_node,ref_N,ref_match,ref_matchedCnt);
	gettimeofday(&end,NULL);
	printf("singe CM-2 %.3f s\n",diffTime(start,end) * 0.001);



	free(sam_node);
	free(ref_node);
	free(ref_match);
	free(sam_match);
	free(ref_matchedCnt);
	free(sam_matchedCnt);
	
	return 0;
}
