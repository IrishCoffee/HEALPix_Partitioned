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

	__global__ 
void test(NODE sam_node[])
{
	int td = blockDim.x * blockIdx.x + threadIdx.x;
	if(td)
		return;
	for(int i = 0; i < 10; ++i)
		printf("%d %lf %lf\n",sam_node[i].pix,sam_node[i].ra,sam_node[i].dec);

}

// return the position of the first element whose pix is greater than key
	__host__ __device__
int begin_index(int key, NODE *node, int N)
{
	int low = 0;
	int high = N - 1;
	while(low <= high)
	{   
		int mid = (low + high) >> 1;
		int midVal = node[mid].pix;
		if(midVal < key)
			low = mid + 1;
		else if(midVal > key)
		{   
			if(mid != 0 && node[mid-1].pix < key)
				return mid;
			if(mid == 0)
				return mid;
			high = mid - 1;
		}   
		else
		{   
			if(mid != 0 && node[mid-1].pix == key)
				high = mid - 1;
			else
				return mid;
		}   
	}   
	return -1; 
}
void singleCM(NODE ref_node[], int ref_N, NODE sam_node[], int sam_N, int sam_match[],int sam_matchedCnt[])
{
	int part_sam_N = 100000000;
	int part_ref_N = 100000000;

	NODE *d_ref_node[2];
	NODE *d_sam_node[2];
	int *d_sam_match[2], *d_sam_matchedCnt[2];

	for(int j = 0; j < sam_N; j += part_sam_N)
	{
		int key = sam_node[j].pix;
		int pos = begin_index(key,ref_node,ref_N);
		printf("key %d from %d \n",key,pos);

		cout << ref_node[pos-1].pix << endl;
		cout << ref_node[pos].pix << endl;
		cout << ref_node[pos+1].pix << endl;
		
		cout << endl;
		int end = j + part_sam_N - 1;
		if(end >= sam_N)
			end = sam_N - 1;

		key = sam_node[end].pix;
		int pos2 = begin_index(key,ref_node,ref_N);
		printf("key %d end %d\n",key,pos2);
		
		cout << ref_node[pos2-1].pix << endl;
		cout << ref_node[pos2].pix << endl;
		cout << ref_node[pos2+1].pix << endl;
		
		cout << "gap " << pos2 - pos << endl;
		cout << "--------------------------------" << endl;
	}

	omp_set_num_threads(2);
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
