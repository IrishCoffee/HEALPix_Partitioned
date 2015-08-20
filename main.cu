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
#include "mpi.h"
#include "tbb/parallel_sort.h"
#include "printResult.h"
#include "kernel_functions.h"
#include "helper_functions.h"
#include "values.h"
#include "worker.h"
#include "master.h"
using namespace std;

int main(int argc, char* argv[])
{
	struct timeval start,end;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Get_processor_name(processor_name,&namelen);

	printf("--------------\nRank %d Processor_name %s\n------------------\n",rank,processor_name);

	if(rank == MASTER_NODE)
	{

		master_allocation();
		gettimeofday(&start,NULL);
		cout << "master before loadfile " << endl;
		master_load_file(argv[2]);
		gettimeofday(&end,NULL);
		printf("master_load_file %.3f s \n", diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		master_getPix();
		gettimeofday(&end,NULL);
		printf("master_getPix %.3f s \n", diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		tbb::parallel_sort(h_sam_node,h_sam_node + sam_N,cmp);
		gettimeofday(&end,NULL);
		printf("master sort %.3f s \n", diffTime(start,end) * 0.001 );

		int *cnt_table = (int *)malloc(sizeof(int) * cntSize);
		memset(cnt_table,0,sizeof(cnt_table));
		for(int i = 0; i < sam_N; ++i)
			cnt_table[h_sam_node[i].pix]++;

		int zero_cnt = 0;
		for(int i = 0; i < cntSize; ++i)
			if(cnt_table[i] == 0)
				zero_cnt++;
		cout << " master zero_cnt " << zero_cnt << endl;

		free(cnt_table);
		master_free();
		MPI_Finalize();
	}
	else
	{
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
		printf("%s worker_memory_allocation %.3f s \n",processor_name, diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		worker_load_file(rank - 1);
		gettimeofday(&end,NULL);
		printf("%s worker_load_file %.3f s \n", processor_name,diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		worker_computeSI(search_radius);
		gettimeofday(&end,NULL);
		printf("%s worker_computeSI %.3f s \n", processor_name,diffTime(start,end) * 0.001 );

		ref_dup_N = 0;
		gettimeofday(&start,NULL);
		worker_duplicateR();
		gettimeofday(&end,NULL);
		printf("%s worker_duplicateR %.3f s \n", processor_name,diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		tbb::parallel_sort(h_ref_dup_node,h_ref_dup_node + ref_dup_N,cmp);
		gettimeofday(&end,NULL);
		printf("%s tbb sort R by key %.3f s \n", processor_name,diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		worker_countR();
		gettimeofday(&end,NULL);
		printf("%s worker_countR %.3f s \n",processor_name, diffTime(start,end) * 0.001 );

		int zeroCnt = 0;
		int sum = 0;
		for(int i = 0; i < cntSize; ++i)
		{
			sum += h_R_cnt[i];
			if(h_R_cnt[i] == 0)
				zeroCnt++;
		}
		printf("sum %d zeroCnt %d percent %.3lf \n",sum,zeroCnt,zeroCnt * 100.0 / cntSize);

		/*
		for(int i = 0; i < 20; ++i)
			printf("%d %.3lf %.3lf %d\n",i,h_ref_dup_node[i].ra,h_ref_dup_node[i].dec,h_ref_dup_node[i].pix);

		for(int i = 0; i < 20; ++i)
			printf("pix %d cnt %d startPos %d\n",i,h_R_cnt[i],h_R_startPos[i]);
*/
		gettimeofday(&start,NULL);
		worker_merge_step1(rank);
		gettimeofday(&end,NULL);
		printf("%s worker_merge %.3f s \n", processor_name,diffTime(start,end) * 0.001 );

		freopen(processor_name,"w",stdout);
		for(int i = 0; i < cntSize; ++i)
			printf("%d\n",h_R_cnt[i]);
		fclose(stdout);

		worker_memory_free();
		MPI_Finalize();
	}
	return 0;
}
