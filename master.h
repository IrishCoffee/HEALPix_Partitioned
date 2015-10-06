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
	free(h_sam_ra);
	free(h_sam_dec);
	free(h_sam_pix);
}

//load sample file, compute HEALPix id for all points
void master_load_file(char *masterFileList)
{
	readSamFile(masterFileList,120);

	/*
	   int offset = 5000000;
	   int cores = 20;
	   int eachN = 240 / cores; //each thread is responsible for 6 files

	   std::ios_base::sync_with_stdio(false);
	   omp_set_num_threads(cores);
#pragma omp parallel
{
int threadId = omp_get_thread_num() % cores;
	//		for(int i = threadId * eachN; i < (threadId + 1) * eachN; ++i)
	for(int i = 0; i < eachN; ++i)
	{
	int k = threadId + i * cores;
	cout << "load  table " << k << endl;
	readSampleData(sam_table[k],h_sam_node + k * offset,offset);
	}
	}
	std::ios_base::sync_with_stdio(true);
	*/
	int SIZE = 1200000000;
	int eachN = SIZE / 12 / 10; 
	for(int k = 0; k < 3; ++k)
	{
#pragma omp parallel for
		for(int i = k * 40; i < (k+1)*40; ++i)
		{

			FILE * fd = fopen(sam_table[i],"r");
			if(fd == NULL)
				printf("load file %s error!\n",sam_table[i]);
			for(int j = 0; j < eachN; ++j)
			{   
				int id = i * eachN + j;
				fscanf(fd,"%lf%lf",&h_sam_node[id].ra,&h_sam_node[id].dec);
			}   
			fclose(fd);
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

void master_toArray()
{
	h_sam_ra = (double *)malloc(sizeof(double) * sam_N);
	h_sam_dec = (double *)malloc(sizeof(double) * sam_N);
	h_sam_pix = (int *)malloc(sizeof(int) * sam_N);
#pragma omp parallel for
	for(int i = 0; i < sam_N; ++i)
	{
		h_sam_ra[i] = h_sam_node[i].ra;
		h_sam_dec[i] = h_sam_node[i].dec;
		h_sam_pix[i] = h_sam_node[i].pix;
	}
	free(h_sam_node);
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
			if(h_sam_pix[j] >= start_pix[i] && h_sam_pix[j] <= end_pix[i])
			{
				cnt_tmp++;
				if(!found_start)
				{
					found_start = true;
					start_pos[i] = j;
				}
			}
			if(h_sam_pix[j] > end_pix[i] || j == sam_N - 1)
			{
				cnt[i] = cnt_tmp;
				break;
			}
		}
	}
	printf("Master finished calcualting the start/end position \n");
	MPI_Request send_request[300];
	MPI_Status send_status[300];
	int send_cnt = 0;
	for(int i = 0; i < worker_N; ++i)
	{
		MPI_Isend(cnt + i,1,MPI_INT,i + 1, 3,MPI_COMM_WORLD,&send_request[send_cnt++]);
		int ite = (int)ceil(cnt[i] * 1.0 / MPI_MESSLEN);
		for(int j = 0; j < ite; ++j)
		{
			int len;
			if(j < ite - 1)
				len = MPI_MESSLEN;
			else 
				len = cnt[i] - j * MPI_MESSLEN;
			//			MPI_Send(h_sam_node + start_pos[i] + MPI_MESSLEN * j,len,mpi_node,i+1,3,MPI_COMM_WORLD);
			//			MPI_Isend(h_sam_node + start_pos[i] + MPI_MESSLEN * j,len,mpi_node,i+1,3,MPI_COMM_WORLD,&send_request[send_cnt++]);
			MPI_Isend(h_sam_ra + start_pos[i] + MPI_MESSLEN * j,len,MPI_DOUBLE,i+1,3,MPI_COMM_WORLD,&send_request[send_cnt++]);
			MPI_Isend(h_sam_dec + start_pos[i] + MPI_MESSLEN * j,len,MPI_DOUBLE,i+1,3,MPI_COMM_WORLD,&send_request[send_cnt++]);
			MPI_Isend(h_sam_pix + start_pos[i] + MPI_MESSLEN * j,len,MPI_INT,i+1,3,MPI_COMM_WORLD,&send_request[send_cnt++]);
			printf("\\\\\\\\ master send to %d iteration %d\n",i + 1,j);
		}
	}
	printf("\\\\\\\\\\\\master send request_cnt %d\n",send_cnt);
	MPI_Waitall(send_cnt,send_request,send_status);
	printf("\\\\\\\ master send request completed\n");
}

#endif
