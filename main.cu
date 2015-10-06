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
#include <fstream>
using namespace std;

int main(int argc, char* argv[])
{
	struct timeval start,end;
	//MPI Communication Initalization
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Get_processor_name(processor_name,&namelen);


	//MPI worker group creation
	MPI_Comm_group(MPI_COMM_WORLD, &entire_group);
	int mem[1] = {0};
	MPI_Group_excl(entire_group,1,mem,&worker_group);
	MPI_Comm_create(MPI_COMM_WORLD,worker_group,&worker_comm);

	printf("--------------\nRank %d Processor_name %s\n------------------\n",rank,processor_name);

	if(rank == MASTER_NODE)
	{
		time_t rawtime;
		time(&rawtime);
		printf("%s starts at %s\n",processor_name,ctime(&rawtime));
		master_allocation();
		gettimeofday(&start,NULL);
		cout << "/////////////// master before loadfile " << endl;
		master_load_file(argv[2]);
		gettimeofday(&end,NULL);
		printf("//////////////// master_load_file %.3f s \n", diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		master_getPix();
		gettimeofday(&end,NULL);
		printf("/////////////// master_getPix %.3f s \n", diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		tbb::parallel_sort(h_sam_node,h_sam_node + sam_N,cmp);
		gettimeofday(&end,NULL);
		printf("///////////// master sort %.3f s \n", diffTime(start,end) * 0.001 );

		gettimeofday(&start,NULL);
		master_toArray();
		gettimeofday(&end,NULL);
		printf("/////////// master transferToArray %.3f s\n",diffTime(start,end) * 0.001);


//		master_free();
		time(&rawtime);
		printf("%s ends at %s\n",processor_name,ctime(&rawtime));

		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&start,NULL);
		master_send_sample(numprocs - 1);
		gettimeofday(&end,NULL);
		printf("//////////// master_send_sample %.3f s\n",diffTime(start,end) * 0.001);
		
		time(&rawtime);
		printf("%s ends at %s\n",processor_name,ctime(&rawtime));

		master_free();
		MPI_Finalize();
		return 0;
	}
	else
	{
		time_t rawtime;
		time(&rawtime);
		printf("%s starts : %s\n",processor_name,ctime(&rawtime));
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
		readRefFile(argv[1],240);
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

		time(&rawtime);
		printf("%s ends : %s\n",processor_name,ctime(&rawtime));

		MPI_Barrier(worker_comm);
		gettimeofday(&start,NULL);
		worker_gather(rank);
		gettimeofday(&end,NULL);
		printf("%s worker_gather %.3f s\n",processor_name,diffTime(start,end) * 0.001);
		
		gettimeofday(&start,NULL);
		worker_merge(rank);
		gettimeofday(&end,NULL);
		printf("%s worker_merge %.3f s\n",processor_name,diffTime(start,end) * 0.001);

		free(h_R_cnt);
		
		time(&rawtime);
		printf("%s after merge : %s\n",processor_name,ctime(&rawtime));
	
		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&start,NULL);
		worker_requestSample(rank);
		gettimeofday(&end,NULL);
		printf("%s worker_request sample %.3f s \n", processor_name,diffTime(start,end) * 0.001 );

		h_ref_dup_ra = (double*)malloc(sizeof(double) * 400000000);
		h_ref_dup_dec = (double*)malloc(sizeof(double) * 400000000);
		h_ref_dup_pix = (int*)malloc(sizeof(int) * 400000000);

		gettimeofday(&start,NULL);
		worker_ownCM(rank);
		gettimeofday(&end,NULL);
		printf("%s worker_ownCM %.3f s \n", processor_name,diffTime(start,end) * 0.001 );
		

		gettimeofday(&start,NULL);
		worker_CM(rank);
		gettimeofday(&end,NULL);
		printf("%s worker_CM %.3f s \n", processor_name,diffTime(start,end) * 0.001 );

		time(&rawtime);
		printf("%s ends : %s\n",processor_name,ctime(&rawtime));
		
		worker_memory_free();
		MPI_Finalize();
		return 0;
	}
}
