#ifndef WORKER_H
#define WORKER_H

#include "helper_functions.h"
#include "values.h"
#include "singleCM_kernel.h"
void mem_allo(int ref_N,int sam_N)
{
	h_ref_ra = (double*) malloc(sizeof(double) * ref_N);
	h_ref_dec = (double*) malloc(sizeof(double) * ref_N);
	h_ref_range = (int*) malloc(sizeof(int) * ref_N * MAX_RANGE_PAIR);

	sam_node_buffer = (PIX_NODE *)malloc(sizeof(PIX_NODE) * sam_N);
}

//load reference and sample file list
void load_file_list(char *ref_list,int ref_file_num,char *sam_list,int sam_file_num)
{
	FILE *fd = fopen(ref_list,"r");
	if(fd == NULL)
	{
		printf("rank-%d load reference file list %s failed\n",rank,ref_list);
		return;
	}
	for(int i = 0; i < ref_file_num; ++i)
		fscanf(fd,"%s",ref_table[i]);
	fclose(fd);

	fd = fopen(sam_list,"r");
	if(fd == NULL)
	{
		printf("rank-%d load sample file list %s failed\n",rank,sam_list);
		return;
	}
	for(int i = 0; i < sam_file_num; ++i)
		fscanf(fd,"%s",sam_table[i]);
	fclose(fd);
}

void load_ref_file(int rank,int ref_file_num,int ref_file_size,int ref_ignore)
{
	int offset = ref_file_size;
	int load_file_num = ref_file_num / numprocs;

#pragma omp parallel for
	for(int i = 0; i < load_file_num; ++i)
	{
		if(readDataFile(ref_table[rank * load_file_num  + i],h_ref_ra + i * offset,h_ref_dec + i * offset,offset,ref_ignore) == -1)
			printf("load file %s error!\n",ref_table[rank * load_file_num + i]);
	}
	return;
}

void load_sam_file(int rank,int sam_file_num,int sam_file_size,int sam_ignore)
{
	int offset = sam_file_size;
	int load_file_num = sam_file_num / numprocs;

#pragma omp parallel for
	for(int i = 0; i < load_file_num; ++i)
	{
		if(readSampleData(sam_table[rank * load_file_num  + i],sam_node_buffer + i * offset,offset,sam_ignore) == -1)
			printf("load file %s error!\n",sam_table[rank * load_file_num + i]);
	}
	return;
}

void computeSI(double search_radius)
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
		//		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		double free = free_mem * 1.0 - GBSize;
		int part_N = min(part_ref_N,(int)floor(free * 1.0 / (8 + 8 + 4 * MAX_RANGE_PAIR)));
		int iteration = (int) ceil(part_ref_N * 1.0 / part_N);

		checkCudaErrors(cudaMalloc(&d_ref_ra[i],sizeof(double) * part_N));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],sizeof(double) * part_N));
		checkCudaErrors(cudaMalloc(&d_ref_range[i],sizeof(int) * part_N * MAX_RANGE_PAIR));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		//		printf("Card %d After malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

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
		//		printf("Card %d before free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaFree(d_ref_ra[i]));
		checkCudaErrors(cudaFree(d_ref_dec[i]));
		checkCudaErrors(cudaFree(d_ref_range[i]));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		//		printf("Card %d After free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);
	}
}

void indexSample()
{
	Healpix_Base healpix_test;
#pragma omp parallel for
	for(int i = 0; i < sam_N; ++i)
	{
		double z = cos(radians(sam_node_buffer[i].dec + 90.0));
		double phi = radians(sam_node_buffer[i].ra);
		sam_node_buffer[i].pix = healpix_test.zphi2pix(z,phi);
	}
}
void count_ref(int rank)
{
	h_R_cnt = (int*)malloc(sizeof(int) * cntSize);
	memset(h_R_cnt,0,sizeof(int) * cntSize);

	for(int i = 0; i < ref_N; ++i)
	{
		int off = i * MAX_RANGE_PAIR;
		for(int j = off;j < (i+1) * MAX_RANGE_PAIR; j += 2)
		{
			if(h_ref_range[j] == -1)
				break;
			for(int k = h_ref_range[j]; k < h_ref_range[j+1]; ++k)
				h_R_cnt[k]++;
		}
	}
}

void cal_samChunk(int rank)
{
	memset(Schunk_size,0,sizeof(Schunk_size));
	memset(S_startPos,-1,sizeof(Schunk_size));
	omp_set_num_threads(worker_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % worker_N;
		int startPix = chunk_start_pix[i];
		int endPix = chunk_end_pix[i];
		for(int j = 0; j < sam_N; ++j)
		{
			if(sam_node_buffer[j].pix >= startPix && S_startPos[i] == -1)
				S_startPos[i] = j;
			if(sam_node_buffer[j].pix >= startPix && sam_node_buffer[j].pix <= endPix)
				Schunk_size[i]++;
		}
	}
	/*
	   for(int i = 0; i < worker_N; ++i)
	   printf("rank-%d sChunk-%d startPos %d size %d\n",rank,i,S_startPos[i],Schunk_size[i]);
	   */
}

void cal_refChunk_size(int rank)
{
	memset(Rchunk_size,0,sizeof(Rchunk_size));
	omp_set_num_threads(worker_N);
#pragma omp parallel
	{
		int chunk_id = omp_get_thread_num() % worker_N;
		for(int i = 0; i < ref_N; ++i)
		{
			int off = i * MAX_RANGE_PAIR;
			for(int j = off; j < (i + 1) * MAX_RANGE_PAIR; j += 2)
			{
				if(h_ref_range[j] == -1)
					break;
				for(int k = h_ref_range[j]; k < h_ref_range[j+1]; ++k)
					if(k >= chunk_start_pix[chunk_id] && k <= chunk_end_pix[chunk_id])
						Rchunk_size[chunk_id]++;
			}
		}
	}
	for(int i = 0; i < worker_N; ++i)
		printf("rank-%d chunk-%d size %d\n",rank,i,Rchunk_size[i]);
}


void worker_gather(int rank)
{
	int worker_root = 0;
	MPI_Status status;
	if(rank == worker_root)
	{
		int *cnt_sum = (int*)malloc(sizeof(int) * cntSize);
		int *cnt_buffer = (int*)malloc(sizeof(int) * cntSize);
		memset(cnt_sum,0,sizeof(int) * cntSize);
		memcpy(cnt_sum,h_R_cnt,sizeof(int) * cntSize);

		printf("rank-%d  before gather\n",rank);
		for(int j = 1; j < numprocs; ++j)
		{
			MPI_Recv(cnt_buffer,cntSize,MPI_INT,j,3,MPI_COMM_WORLD,&status);
			printf("rank-%d recv from rank-%d cntChunk\n",rank,j);
#pragma omp parallel for
			for(int i = 0; i < cntSize; ++i)
				cnt_sum[i] += cnt_buffer[i];
		}
		printf("rank-%d  after gather\n",rank);
		free(cnt_buffer);

		long long sum = 0;
		for(int i = 0; i < cntSize; ++i)
			sum += cnt_sum[i];
		double avg_sum = sum * 1.0 / worker_N;
		int cur_sum = 0;
		int prev_sid = 0;
		int cal_cnt = 0;
		cout << "avg sum " << avg_sum << endl;
		for(int i = 0; i < cntSize; ++i)
		{
			if(cur_sum + cnt_sum[i] <= avg_sum)
				cur_sum += cnt_sum[i];
			else
			{
				chunk_start_pix[cal_cnt] = prev_sid;
				chunk_end_pix[cal_cnt] = i - 1;
				printf("[%d,%d] -> %d\n",prev_sid,i - 1,cur_sum);
				cal_cnt++;
				prev_sid = i;
				cur_sum = cnt_sum[i];
				if(cal_cnt == worker_N - 1)
					break;
			}
		}
		chunk_start_pix[cal_cnt] = prev_sid;
		chunk_end_pix[cal_cnt] = cntSize - 1;
		printf("[%d,%d]\n",prev_sid,cntSize - 1);
		MPI_Bcast(chunk_start_pix,worker_N,MPI_INT,worker_root,MPI_COMM_WORLD);
		MPI_Bcast(chunk_end_pix,worker_N,MPI_INT,worker_root,MPI_COMM_WORLD);
		free(cnt_sum);
	}
	else
	{
		printf("rank-%d  before gather\n",rank);
		MPI_Send(h_R_cnt,cntSize,MPI_INT,worker_root,3,MPI_COMM_WORLD);
		printf("rank-%d  after gather\n",rank);
		MPI_Bcast(chunk_start_pix,worker_N,MPI_INT,worker_root,MPI_COMM_WORLD);
		MPI_Bcast(chunk_end_pix,worker_N,MPI_INT,worker_root,MPI_COMM_WORLD);
	}
	printf("rank-%d finished broadcast\n",rank);
	memset(Rchunk_size,0,sizeof(Rchunk_size));
#pragma omp parallel for
	for(int i = 0; i < worker_N; ++i)
	{
		int lower = chunk_start_pix[i];
		int upper = chunk_end_pix[i];
		for(int j = lower; j <= upper; ++j)
			Rchunk_size[i] += h_R_cnt[j];
	}
	//	for(int i = 0; i < worker_N; ++i)
	//		printf("rank-%d chunk-%d size-%d\n",rank,i,Rchunk_size[i]);
	free(h_R_cnt);
}

void transfer_ref(int rank,int dest_rank,PIX_NODE *ref_chunk,int chunk_N)
{
	int lower = chunk_start_pix[dest_rank];
	int upper = chunk_end_pix[dest_rank];

	int cnt = 0;

	for(int i = 0; i < ref_N; ++i)
	{
		int off = i * MAX_RANGE_PAIR;
		for(int j = off;j < (i+1) * MAX_RANGE_PAIR; j += 2)
		{
			if(h_ref_range[j] == -1)
				break;
			for(int k = h_ref_range[j]; k < h_ref_range[j+1]; ++k)
			{
				if(k >= lower && k <= upper)
				{
					ref_chunk[cnt].ra = h_ref_ra[i];
					ref_chunk[cnt].dec = h_ref_dec[i];
					ref_chunk[cnt].pix = k;
					cnt++;
				}
			}
		}
	}
	printf("rank-%d transfer_ref chunk-%d size-%d \n",rank,dest_rank,cnt);
}

void sr_R_chunk(int send_recv,int dest,int source,PIX_NODE *buffer,int size)
{
	MPI_Status status;
	int ite = (size - 1) / MPI_MESSLEN + 1;
	for(int i = 0; i < ite; ++i)
	{
		int sr_N;
		if(i == ite - 1)
			sr_N = size - i * MPI_MESSLEN;
		else
			sr_N = MPI_MESSLEN;
		if(send_recv == 1) // recv
		{
			MPI_Recv(buffer + MPI_MESSLEN * i,sr_N,mpi_node,source,3,MPI_COMM_WORLD,&status);
			printf("rank-%d recv from rank-%d chunk-%d\n",dest,source,i);
		}
		else
		{
			MPI_Send(buffer + MPI_MESSLEN * i,sr_N,mpi_node,dest,3,MPI_COMM_WORLD);
			printf("rank-%d sent to rank-%d chunk-%d\n",source,dest,i);
		}
	}
}

//Isend Irecv
void Isr_R_chunk(int send_recv,int dest,int source,PIX_NODE *buffer,int size)
{
	MPI_Status status;
	int ite = (size - 1) / MPI_MESSLEN + 1;
	for(int i = 0; i < ite; ++i)
	{
		int sr_N;
		if(i == ite - 1)
			sr_N = size - i * MPI_MESSLEN;
		else
			sr_N = MPI_MESSLEN;
		if(send_recv == 1) // recv
		{
			MPI_Recv(buffer + MPI_MESSLEN * i,sr_N,mpi_node,source,3,MPI_COMM_WORLD,&status);
			//		MPI_Irecv(buffer + MPI_MESSLEN * i,sr_N,mpi_node,source,3,MPI_COMM_WORLD,&recv_ref_request[recv_ref_cnt++]);
			printf("rank-%d recv from rank-%d chunk-%d\n",dest,source,i);
		}
		else
		{
			//	MPI_Send(buffer + MPI_MESSLEN * i,sr_N,mpi_node,dest,3,MPI_COMM_WORLD);
			MPI_Isend(buffer + MPI_MESSLEN * i,sr_N,mpi_node,dest,3,MPI_COMM_WORLD,&send_ref_request[send_ref_cnt++]);	
			printf("rank-%d sent to rank-%d chunk-%d\n",source,dest,i);
		}
	}
}

void redistribute_R(int rank)
{
	send_ref_cnt = 0;
	recv_ref_cnt = 0;
	MPI_Status status;
	int each_chunk_N[worker_N];
	each_chunk_N[rank] = Rchunk_size[rank];

	omp_set_num_threads(worker_N);
	int offset = 0;
	for(int i = 0; i < worker_N; ++i)
	{
		if(i == rank)
		{
			for(int j = 0; j < worker_N; ++j)
			{
				if(j != rank)
					MPI_Recv(each_chunk_N + j,1,MPI_INT,j,3,MPI_COMM_WORLD,&status);
			}
			ref_CM_N = 0;
			for(int j = 0; j < worker_N; ++j)
				ref_CM_N += each_chunk_N[j];
			h_ref_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * ref_CM_N);
			cout << "rank-" << rank << " ref_CM_N " << ref_CM_N << endl;
			for(int j = 0; j < worker_N; ++j)
			{
				if(j == rank)
					continue;
				Isr_R_chunk(1,rank,j,h_ref_node + offset,each_chunk_N[j]);
				offset += each_chunk_N[j];
			}
		}
		else
		{
			MPI_Send(Rchunk_size + i,1,MPI_INT,i,3,MPI_COMM_WORLD);
			PIX_NODE *ref_buffer = (PIX_NODE *)malloc(sizeof(PIX_NODE) * Rchunk_size[i]);
			struct timeval start,end;
			gettimeofday(&start,NULL);
			transfer_ref(rank,i,ref_buffer,Rchunk_size[i]);
			gettimeofday(&end,NULL);
			printf("rank-%d transfer chunk-%d costs %.3f \n",rank,i,diffTime(start,end) * 0.001);
			Isr_R_chunk(0,i,rank,ref_buffer,Rchunk_size[i]);
			free(ref_buffer);
		}
	}
	ref_CM_N -= Rchunk_size[rank];
	//	PIX_NODE *ref_buffer = (PIX_NODE *)malloc(sizeof(PIX_NODE) * Rchunk_size[rank]);
	//	transfer_ref(rank,rank,ref_buffer,Rchunk_size[rank]);
	//	printf("rank-%d ref_offset %d R_chunk_size %d\n",rank,offset,Rchunk_size[rank]);
	//	memcpy(h_ref_node + offset,ref_buffer,Rchunk_size[rank]);
	//	free(ref_buffer);
	free(h_ref_ra);
	free(h_ref_dec);
	free(h_ref_range);
	printf("rank-%d send_ref_cnt %d recv_ref_cnt %d\n",rank,send_ref_cnt,recv_ref_cnt);
	//	MPI_Waitall(send_ref_cnt,send_ref_request,send_ref_status);
	//	MPI_Waitall(recv_ref_cnt,recv_ref_request,recv_ref_status);
	for(int i = 0; i < send_ref_cnt; ++i)
	{
		int index;
		MPI_Status status;
		MPI_Waitany(send_ref_cnt,send_ref_request,&index,&status);
		printf("rank-%d finished sent to rank-%d",rank,status.MPI_SOURCE);
	}
	/*
	   for(int i = 0; i < recv_ref_cnt; ++i)
	   {
	   int index;
	   MPI_Status status;
	   MPI_Waitany(recv_ref_cnt,recv_ref_request,&index,&status);
	   printf("rank-%d finished recv from rank-%d",rank,status.MPI_SOURCE);
	   }
	   */
	printf("rank-%d finished MPI_Waitall\n",rank);
}
void redistribute_R_bak(int rank)
{
	MPI_Status status;
	int each_R_chunk_N[worker_N];
	each_R_chunk_N[rank] = Rchunk_size[rank];

	omp_set_num_threads(worker_N);
	int offset = 0;
	for(int i = 0; i < worker_N; ++i)
	{
		if(i == rank)
		{
			for(int j = 0; j < worker_N; ++j)
			{
				if(j != rank)
					MPI_Recv(each_R_chunk_N + j,1,MPI_INT,j,3,MPI_COMM_WORLD,&status);
			}
			ref_CM_N = 0;
			for(int j = 0; j < worker_N; ++j)
				ref_CM_N += each_R_chunk_N[j];
			printf("rank-%d ref_CM_N %d sam_CM_N %d\n",rank,ref_CM_N,sam_CM_N);
		}
		else
			MPI_Send(Rchunk_size + i,1,MPI_INT,i,3,MPI_COMM_WORLD);
	}
	h_ref_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * ref_CM_N);
	for(int i = 0; i < worker_N - 1; ++i)
	{
		int commID = request_node[rank][i];
		int R_send_amount = Rchunk_size[commID];
		int R_recv_amount = each_R_chunk_N[commID];

		int R_send_ite = (R_send_amount - 1) / MPI_MESSLEN + 1;
		int R_recv_ite = (R_recv_amount - 1) / MPI_MESSLEN + 1;

		PIX_NODE *ref_buffer = (PIX_NODE*)malloc(sizeof(PIX_NODE) * R_send_amount);
		transfer_ref(rank,commID,ref_buffer,Rchunk_size[commID]);

		int RstartPos = 0, SstartPos = 0;
		for(int j = 0; j < commID; ++j)
		{
			if(j == rank)
				continue;
			RstartPos += each_R_chunk_N[j];
		}

		for(int kk = 0; kk < 2; ++kk)
		{
			if((kk == 0 && commID > rank) || (kk == 1 && commID < rank))
			{
				for(int j = 0; j < R_send_ite; ++j)
				{
					int cur_N;
					if(j == R_send_ite - 1)
						cur_N = R_send_amount - j * MPI_MESSLEN;
					else
						cur_N = MPI_MESSLEN;

					MPI_Send(ref_buffer + j * MPI_MESSLEN,cur_N,mpi_node,commID,3,MPI_COMM_WORLD);
					printf("rank-%d send to rank-%d R_chunk-%d\n",rank,commID,j);
				}
			}
			else if((kk == 0 && commID < rank) || (kk == 1 && commID > rank))
			{
				for(int j = 0; j < R_recv_ite; ++j)
				{
					int cur_N;
					if(j == R_recv_ite - 1)
						cur_N = R_recv_amount - j * MPI_MESSLEN;
					else
						cur_N = MPI_MESSLEN;

					MPI_Recv(h_ref_node + RstartPos + j * MPI_MESSLEN,cur_N,mpi_node,commID,3,MPI_COMM_WORLD,&status);
					printf("rank-%d recv from rank-%d R_chunk-%d\n",rank,commID,j);
				}
			}
		}
		free(ref_buffer);
	}
	ref_CM_N -= Rchunk_size[rank];
	free(h_ref_ra);
	free(h_ref_dec);
	free(h_ref_range);
}


void redistribute_S(int rank)
{
	MPI_Status status;
	int each_chunk_N[worker_N];
	each_chunk_N[rank] = Schunk_size[rank];

	int offset = 0;
	for(int i = 0; i < worker_N; ++i)
	{
		if(i == rank)
		{
			for(int j = 0; j < worker_N; ++j)
			{
				if(j != rank)
					MPI_Recv(each_chunk_N + j,1,MPI_INT,j,3,MPI_COMM_WORLD,&status);
			}
			sam_CM_N = 0;
			for(int j = 0; j < worker_N; ++j)
				sam_CM_N += each_chunk_N[j];

			h_sam_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * sam_CM_N);
			cout << "rank-" << rank << " sam_CM_N " << sam_CM_N << endl;

			for(int j = 0; j < worker_N; ++j)
			{
				if(j == rank)
					continue;
				sr_R_chunk(1,rank,j,h_sam_node + offset,each_chunk_N[j]);
				offset += each_chunk_N[j];
			}
		}
		else
		{
			MPI_Send(Schunk_size + i,1,MPI_INT,i,3,MPI_COMM_WORLD);
			sr_R_chunk(0,i,rank,sam_node_buffer + S_startPos[i],Schunk_size[i]);
		}
	}
	sam_CM_N -= Schunk_size[rank];
	//	memcpy(h_sam_node + offset,sam_node_buffer + S_startPos[rank],Schunk_size[rank]);
	free(sam_node_buffer);
}

//void singleCM(PIX_NODE h_ref_node[],int ref_CM_N,PIX_NODE h_sam_node[],int sam_CM_N,int rank)
void singleCM(int rank)
{
	printf("rank-%d ref_CM_N %d sam_CM_N %d\n",rank,ref_CM_N,sam_CM_N);
	int *h_sam_matchedCnt = (int*)malloc(sizeof(int) * sam_CM_N);
	int *h_sam_match = (int*)malloc(sizeof(int) * sam_CM_N * 5);
	//the maximum number of sample points that can be matched each time by each card
	int part_sam_N = 10000000;
	int part_ref_N = 10 * part_sam_N;

	PIX_NODE *d_ref_node[GPU_N];
	PIX_NODE *d_sam_node[GPU_N];
	int *d_sam_match[GPU_N], *d_sam_matchedCnt[GPU_N];

	int chunk_N = (int)ceil(sam_CM_N * 1.0 / part_sam_N);
	int chunk_id = 0;

	omp_set_num_threads(GPU_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("rank-%d Card %d before malloc %.2lf GB, total memory %.2lf GB\n",rank,i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);


		checkCudaErrors(cudaMalloc(&d_ref_node[i],sizeof(PIX_NODE) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_sam_node[i],sizeof(PIX_NODE) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_match[i],sizeof(int) * part_sam_N  * 5));
		checkCudaErrors(cudaMalloc(&d_sam_matchedCnt[i],sizeof(int) * part_sam_N));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("rank-%d Card %d after malloc %.2lf GB, total memory %.2lf GB\n",rank,i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		while(chunk_id < chunk_N)
			//the total number of sample points processed by this card
		{
#pragma omp atomic
			chunk_id++;

			int cur_sam_N;
			if(chunk_id == chunk_N) // the last round
				cur_sam_N = sam_CM_N - (chunk_id - 1) * part_sam_N;
			else
				cur_sam_N = part_sam_N;

			int start_sam_pos = (chunk_id - 1) * part_sam_N;
			int end_sam_pos = start_sam_pos + cur_sam_N - 1;

			int start_pix = h_sam_node[start_sam_pos].pix;
			int end_pix = h_sam_node[end_sam_pos].pix;

			int start_ref_pos;
			if(start_pix == 0)
				start_ref_pos = 0;
			else
				start_ref_pos = binary_search(start_pix - 1,h_ref_node,ref_CM_N);
			//				start_ref_pos = get_start(start_pix,h_ref_node,ref_CM_N);

			if(start_ref_pos == -1)
				continue;
			int end_ref_pos = binary_search(end_pix,h_ref_node,ref_CM_N) - 1;
			if(end_ref_pos == -2)
				end_ref_pos = ref_CM_N - 1;
			int cur_ref_N = end_ref_pos - start_ref_pos + 1;

			dim3 block(block_size);
			dim3 grid(min(65536,(int)ceil(cur_sam_N * 1.0 / block.x)));

			if(cur_ref_N == 0)
				continue;

			printf("\n\nrank-%d Card %d chunk-%d\n",rank,i,chunk_id - 1);
			//					printf("block.x %d grid.x %d\n",block.x,grid.x);
			printf("rank-%d start_sam_pos %d start_sam_pix %d end_sam_pos %d end_sam_pix %d sam_N %d\n",rank,start_sam_pos,start_pix,end_sam_pos,end_pix,cur_sam_N);
			printf("rank-%d start_ref_pos %d start_ref_pix %d end_ref_pos %d end_ref_pix %d ref_N %d\n",rank,start_ref_pos,h_ref_node[start_ref_pos].pix,end_ref_pos,h_ref_node[end_ref_pos].pix,cur_ref_N);
			checkCudaErrors(cudaMemset(d_sam_matchedCnt[i],0,sizeof(int) * part_sam_N));
			checkCudaErrors(cudaMemcpy(d_sam_node[i],h_sam_node + start_sam_pos,cur_sam_N * sizeof(PIX_NODE),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_node[i],h_ref_node + start_ref_pos,cur_ref_N * sizeof(PIX_NODE), cudaMemcpyHostToDevice));
			kernel_singleCM<<<grid,block>>>(d_ref_node[i],cur_ref_N,d_sam_node[i],cur_sam_N,d_sam_match[i],d_sam_matchedCnt[i],start_ref_pos,start_sam_pos);
			checkCudaErrors(cudaMemcpy(h_sam_matchedCnt + start_sam_pos,d_sam_matchedCnt[i],cur_sam_N * sizeof(int),cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(h_sam_match + start_sam_pos * 5,d_sam_match[i],cur_sam_N * 5 * sizeof(int),cudaMemcpyDeviceToHost));
		}

		checkCudaErrors(cudaFree(d_sam_matchedCnt[i]));
		checkCudaErrors(cudaFree(d_sam_match[i]));
		checkCudaErrors(cudaFree(d_ref_node[i]));
		checkCudaErrors(cudaFree(d_sam_node[i]));

	}
	unsigned long long sum = 0;
	int cnt[1000];
	memset(cnt,0,sizeof(cnt));
	for(int i = sam_CM_N - 1; i >= 0; --i)
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
	cout << "rank-" << rank << " sum " << sum << endl;
	cout << "rank-" << rank << " ave " << sum * 1.0 / sam_CM_N << endl;
}

#endif
