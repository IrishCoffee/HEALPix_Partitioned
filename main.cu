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

	//Register new data type
	MPI_Datatype old_types[3];
	MPI_Aint indices[3];
	int blocklens[3];
	blocklens[0] = 1;
	blocklens[1] = 1;
	blocklens[2] = 1;
	old_types[0] = MPI_DOUBLE;
	old_types[1] = MPI_DOUBLE;
	old_types[2] = MPI_INT;

	indices[0] = 0;
	indices[1] = sizeof(double);
	indices[2] = 2 * sizeof(double);

	MPI_Type_struct(3,blocklens,indices,old_types,&mpi_node);
	MPI_Type_commit(&mpi_node);


//	freopen(processor_name,"w",stdout);
	char *referenceTable = argv[1];
	int ref_file_num = atoi(argv[2]);
	int ref_file_size = atoi(argv[3]);
	int ref_file_ignore = atoi(argv[4]);
	char *sampleTable = argv[5];
	int sam_file_num = atoi(argv[6]);
	int sam_file_size = atoi(argv[7]);
	int sam_file_ignore = atoi(argv[8]);

	ref_N = ref_file_num * ref_file_size / numprocs;
	sam_N = sam_file_num * sam_file_size / numprocs;

	cout << "ref_N " << sam_N << endl;

	time_t rawtime;
	time(&rawtime);
	printf("--------------\nRank %d Processor_name %s\n------------------\n",rank,processor_name);
	printf("%s starts at %s\n",processor_name,ctime(&rawtime));

	mem_allo(ref_file_num * ref_file_size / numprocs, sam_file_num * sam_file_size / numprocs);
	load_file_list(referenceTable,ref_file_num,sampleTable,sam_file_num);

	gettimeofday(&start,NULL);
	load_ref_file(rank,ref_file_num,ref_file_size,ref_file_ignore);
	gettimeofday(&end,NULL);
	printf("rank-%d load_ref_file %.3f s\n",rank,diffTime(start,end) * 0.001);

	gettimeofday(&start,NULL);
	load_sam_file(rank,sam_file_num,sam_file_size,sam_file_ignore);
	gettimeofday(&end,NULL);
	printf("rank-%d load_sam_file %.3f s\n",rank,diffTime(start,end) * 0.001);

	gettimeofday(&start,NULL);
	computeSI(search_radius);
	gettimeofday(&end,NULL);
	printf("rank-%d computeSI %.3f s\n",rank,diffTime(start,end) * 0.001);

	gettimeofday(&start,NULL);
	indexSample();
	gettimeofday(&end,NULL);
	printf("rank-%d indexSample %.3f s\n",rank,diffTime(start,end) * 0.001);

	gettimeofday(&start,NULL);
	tbb::parallel_sort(h_sam_node,h_sam_node + sam_N,cmp);
	gettimeofday(&end,NULL);
	printf("rank-%d sort sample %.3f s\n",rank,diffTime(start,end) * 0.001);

	gettimeofday(&start,NULL);
	count_ref(rank);
	gettimeofday(&end,NULL);
	printf("rank-%d count_ref %.3f s\n",rank,diffTime(start,end) * 0.001);

	gettimeofday(&start,NULL);
	worker_gather(rank);
	gettimeofday(&end,NULL);
	printf("rank-%d worker_gather %.3f s\n",rank,diffTime(start,end) * 0.001);

	gettimeofday(&start,NULL);
	redistribute_R(rank);
	gettimeofday(&end,NULL);
	printf("rank-%d redistribute_R %.3f s\n",rank,diffTime(start,end) * 0.001);

	time(&rawtime);
	printf("%s ends at %s\n",processor_name,ctime(&rawtime));

}
