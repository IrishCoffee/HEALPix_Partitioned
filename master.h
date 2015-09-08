#ifndef MASTER_H
#define MASTER_H

#include "values.h"
#include "helper_functions.h"

void master_allocation()
{
	h_sam_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * sam_N);
}

void master_free()
{
	free(h_sam_node);
}

//load sample file, compute HEALPix id for all points
void master_load_file(char *masterFileList)
{
	readSamFile(masterFileList,240);

	int offset = 5000000;
	int cores = 40;
	int eachN = 240 / cores; //each thread is responsible for 6 files

	omp_set_num_threads(cores);
#pragma omp parallel
	{
		int threadId = omp_get_thread_num() % cores;
		//		for(int i = threadId * eachN; i < (threadId + 1) * eachN; ++i)
		for(int i = 0; i < eachN; ++i)
		{
			int k = threadId + i * cores;
			readSampleData(sam_table[k],h_sam_node + k * offset,offset);
		}
	}
}

void master_getPix()
{
	Healpix_Base healpix_test;
#pragma omp parallel for
	for(int i = 0; i < sam_N; ++i)
	{   
		double z = cos(radians(h_sam_node[i].dec + 90.0));
		double phi = radians(h_sam_node[i].ra);
		h_sam_node[i].pix = healpix_test.zphi2pix(z,phi);
	}   
}

void master_send_sample(int worker_N)
{
	MPI_Status status;
	int start_pix[worker_N],end_pix[worker_N];
	int start_pos[worker_N];
	int cnt[worker_N];
	for(int i = 0; i < worker_N; ++i)
	{
		MPI_Recv(start_pix + i,1,MPI_INT,i + 1,3,MPI_COMM_WORLD,&status);
		printf("//////// master recv from %d start_pix %d\n",status.MPI_SOURCE,start_pix[i]);
		MPI_Recv(end_pix + i,1,MPI_INT,i + 1,3,MPI_COMM_WORLD,&status);
		printf("//////// master recv from %d end_pix %d\n",status.MPI_SOURCE,end_pix[i]);
	}
	omp_set_num_threads(worker_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % worker_N;
		int cnt_tmp = 0;
		bool found_start = false;
		for(int j = 0; j < sam_N; ++j)
		{
			if(h_sam_node[j].pix >= start_pix[i] && h_sam_node[j].pix <= end_pix[i])
			{
				cnt_tmp++;
				if(!found_start)
				{
					found_start = true;
					start_pos[i] = j;
				}
			}
			if(h_sam_node[j].pix > end_pix[i] || j == sam_N - 1)
			{
				cnt[i] = cnt_tmp;
				break;
			}
		}
	}
	cout << 0 << " " << start_pos[0] << endl;
	cout << 1 << " " << start_pos[1] << endl;
	for(int i = 0; i < worker_N; ++i)
	{
		MPI_Send(cnt + i,1,MPI_INT,i + 1, 3,MPI_COMM_WORLD);
		int ite = (int)ceil(cnt[i] * 1.0 / MPI_MESSLEN);
		for(int j = 0; j < ite; ++j)
		{
			int len;
			if(j < ite - 1)
				len = MPI_MESSLEN;
			else 
				len = cnt[i] - j * MPI_MESSLEN;
			printf("\\\\\\\\ master send to %d iteration %d\n",rank,j);
			MPI_Send(h_sam_node + start_pos[i] + MPI_MESSLEN * j,len,mpi_node,i+1,3,MPI_COMM_WORLD);
		}
	}
	for(int i = 0; i < 10; ++i)
		printf("master i %d pix %d ra %.6lf dec %.6lf\n",i,h_sam_node[i].pix,h_sam_node[i].ra,h_sam_node[i].dec);
	printf("\\\\\\\\\\\\ \n");
	for(int i = start_pos[1]; i < start_pos[1] + 10; ++i)
		printf("master i %d pix %d ra %.6lf dec %.6lf\n",i,h_sam_node[i].pix,h_sam_node[i].ra,h_sam_node[i].dec);
}

#endif
