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
using namespace std;

int ref_line[20];
char ref_file[20][16];

int sam_line[20];
char sam_file[20][16];

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
}
