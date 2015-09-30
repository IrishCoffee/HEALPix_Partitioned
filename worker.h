#ifndef WORKER_H
#define WORKER_H

#include "helper_functions.h"
#include "values.h"
#include "singleCM_kernel.h"

void worker_memory_allocation()
{
	h_ref_ra = (double*) malloc(sizeof(double) * ref_N);
	h_ref_dec = (double*) malloc(sizeof(double) * ref_N);
	h_ref_range = (int*) malloc(sizeof(int) * ref_N * MAX_RANGE_PAIR);
}

void worker_memory_free()
{
	free(h_ref_dup_node);
	free(h_worker_sam_ra);
	free(h_worker_sam_dec);
	free(h_worker_sam_pix);
	free(h_ref_dup_ra);
	free(h_ref_dup_dec);
	free(h_ref_dup_pix);
}

void worker_load_file(int workerID)
{
	char table[20][64];
	strcpy(table[0],ref_table[workerID * 2]);
	int offset = ref_N / 2 / 20;

	omp_set_num_threads(20);
#pragma omp parallel 
	{
		int i = omp_get_thread_num() % 20;
		strcpy(table[i],ref_table[workerID * 2]);
		int len = strlen(table[i]);
		table[i][len-1] += i;
		if(readDataFile(table[i],h_ref_ra + i * offset,h_ref_dec + i * offset,offset) == -1)
			printf("load file %s error!\n",table[i]);
		//load the second directory
		strcpy(table[i],ref_table[workerID * 2 + 1]);
		len = strlen(table[i]);
		table[i][len-1] += i;
		if(readDataFile(table[i],h_ref_ra + (i + 20) * offset,h_ref_dec + (i + 20) * offset,offset) == -1)
			printf("load file %s error!\n",table[i]);
	}
	return;
}

void worker_computeSI(double search_radius)
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
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		double free = free_mem * 1.0 - GBSize;
		int part_N = min(part_ref_N,(int)floor(free * 1.0 / (8 + 8 + 4 * MAX_RANGE_PAIR)));
		int iteration = (int) ceil(part_ref_N * 1.0 / part_N);

		checkCudaErrors(cudaMalloc(&d_ref_ra[i],sizeof(double) * part_N));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],sizeof(double) * part_N));
		checkCudaErrors(cudaMalloc(&d_ref_range[i],sizeof(int) * part_N * MAX_RANGE_PAIR));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

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
		printf("Card %d before free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaFree(d_ref_ra[i]));
		checkCudaErrors(cudaFree(d_ref_dec[i]));
		checkCudaErrors(cudaFree(d_ref_range[i]));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d After free %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);
	}
}

void worker_duplicateR()
{
	h_ref_dup_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * ref_N * 10);

	printf("before duplicate ref_dup_N %d\n",ref_dup_N);
	//#pragma omp parallel for
	for(int i = 0; i < ref_N; ++i)
	{
		int off = i * MAX_RANGE_PAIR;
		for(int j = off;j < (i+1) * MAX_RANGE_PAIR; j += 2)
		{
			if(h_ref_range[j] == -1)
				break;
			for(int k = h_ref_range[j]; k < h_ref_range[j+1]; ++k)
			{
				h_ref_dup_node[ref_dup_N].ra = h_ref_ra[i];
				h_ref_dup_node[ref_dup_N].dec = h_ref_dec[i];
				h_ref_dup_node[ref_dup_N].pix = k;
				ref_dup_N++;
			}
		}
	}
	free(h_ref_ra);
	free(h_ref_dec);
	free(h_ref_range);
	printf("ref_dup_N % d\n",ref_dup_N);
}

void worker_countR()
{
	h_R_cnt = (int*)malloc(sizeof(int) * cntSize);
	memset(h_R_cnt,0,sizeof(int) * cntSize);
	int pix;
	for(int i = 0; i < ref_dup_N; ++i)
	{
		pix = h_ref_dup_node[i].pix;
		h_R_cnt[pix]++;
	}
}
void worker_toArray(int start_pos,int N)
{
#pragma omp parallel for
	for(int i = 0; i < N; ++i)
	{
		h_ref_dup_ra[i] = h_ref_dup_node[i + start_pos].ra;
		h_ref_dup_dec[i] = h_ref_dup_node[i + start_pos].dec;
		h_ref_dup_pix[i] = h_ref_dup_node[i + start_pos].pix;
	}
}

void worker_merge(int rank)
{
	chunk_start_pix[0] = 0;
	chunk_start_pix[1] = 307592347;
	chunk_start_pix[2] = 379936659;
	chunk_start_pix[3] = 492136207;
	chunk_start_pix[4] = 631888855;
	chunk_start_pix[5] = 703462320;

	chunk_end_pix[0] = 307592346;
	chunk_end_pix[1] = 379936658;
	chunk_end_pix[2] = 492136206;
	chunk_end_pix[3] = 631888854;
	chunk_end_pix[4] = 703462319;
	chunk_end_pix[5] = 805306367;
	int start_pos[6];
	int end_pos[6];
	omp_set_num_threads(worker_N * 2);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % (worker_N * 2);
		int start_pix,end_pix;
		if(i < 6)
		{
			start_pix = chunk_start_pix[i];
			start_pos[i] = binary_search(start_pix - 1,h_ref_dup_node,ref_dup_N);
		}
		else
		{
			end_pix = chunk_end_pix[i % worker_N];
			end_pos[i % worker_N] = binary_search(end_pix,h_ref_dup_node,ref_dup_N);
			if(end_pos[i % worker_N] == -1)
				end_pos[i % worker_N] = ref_dup_N;
		}
	}

	omp_set_num_threads(worker_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % worker_N;
		pix_chunk_startPos[i] = start_pos[i];
		pix_chunk_cnt[i] = end_pos[i] - start_pos[i];
	}

	for(int i = 0; i < 6; ++i)
		printf("rank-%d pixChunk-%d cnt %d startPos %d\n",rank,i,pix_chunk_cnt[i],pix_chunk_startPos[i]);
	return;

}
void worker_requestSample(int rank)
{
	printf("rank-%d prepare to send start/end pix to master\n",rank);
	MPI_Status status;

	MPI_Send(&chunk_start_pix[rank-1],1,MPI_INT,MASTER_NODE,3,MPI_COMM_WORLD);
	MPI_Send(&chunk_end_pix[rank-1],1,MPI_INT,MASTER_NODE,3,MPI_COMM_WORLD);
	printf("rank-%d send start and end to master\n",rank);

	MPI_Recv(&worker_sam_N,1,MPI_INT,MASTER_NODE,3,MPI_COMM_WORLD,&status);
	printf("rank-%d request sample amount %d\n",rank,worker_sam_N);

	h_worker_sam_ra = (double *)malloc(sizeof(double) * worker_sam_N);
	h_worker_sam_dec = (double *)malloc(sizeof(double) * worker_sam_N);
	h_worker_sam_pix = (int *)malloc(sizeof(int) * worker_sam_N);

	int ite = (int)ceil(worker_sam_N * 1.0 / MPI_MESSLEN);
	int request_cnt = 0;
	MPI_Request recv_request[ite];
	MPI_Status recv_status[ite];

	for(int i = 0; i < ite; ++i)
	{
		int len;
		if(i < ite - 1)
			len = MPI_MESSLEN;
		else
			len = worker_sam_N - i * MPI_MESSLEN;
		MPI_Recv(h_worker_sam_ra + i * MPI_MESSLEN,len,MPI_DOUBLE,MASTER_NODE,3,MPI_COMM_WORLD,&status);
		MPI_Recv(h_worker_sam_dec + i * MPI_MESSLEN,len,MPI_DOUBLE,MASTER_NODE,3,MPI_COMM_WORLD,&status);
		MPI_Recv(h_worker_sam_pix + i * MPI_MESSLEN,len,MPI_INT,MASTER_NODE,3,MPI_COMM_WORLD,&status);
		printf("rank-%d recv sample chunk %d\n",rank,i);
	}

	printf("rank-%d recv_cnt %d\n",rank,request_cnt);
	//	MPI_Waitall(request_cnt,recv_request,recv_status);
	printf("rank-%d recv completed\n",rank);
}

void singleCM(double *ref_ra,double *ref_dec,int *ref_pix,double *sam_ra,double *sam_dec,int *sam_pix,int ref_N,int sam_N,int *h_sam_match,int *h_sam_matchedCnt)
{
	//the maximum number of sample points that can be matched each time by each card
	int part_sam_N = 25000000;
	int part_ref_N = 8 * part_sam_N;


	double *d_ref_ra[GPU_N];
	double *d_ref_dec[GPU_N];
	int *d_ref_pix[GPU_N];

	double *d_sam_ra[GPU_N];
	double *d_sam_dec[GPU_N];
	int *d_sam_pix[GPU_N];

	int *d_sam_match[GPU_N], *d_sam_matchedCnt[GPU_N];

	int chunk_N = (int)ceil(sam_N * 1.0 / part_sam_N);
	int chunk_id = 0;

	omp_set_num_threads(GPU_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % GPU_N;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaDeviceReset());

		size_t free_mem,total_mem;
		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d before malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);


		checkCudaErrors(cudaMalloc(&d_ref_ra[i],sizeof(double) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],sizeof(double) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_pix[i],sizeof(int) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_sam_ra[i],sizeof(double) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_dec[i],sizeof(double) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_pix[i],sizeof(int) * part_sam_N));


		checkCudaErrors(cudaMalloc(&d_sam_match[i],sizeof(int) * part_sam_N  * 5));
		checkCudaErrors(cudaMalloc(&d_sam_matchedCnt[i],sizeof(int) * part_sam_N));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		printf("Card %d after malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		while(chunk_id < chunk_N)
			//the total number of sample points processed by this card
		{
#pragma omp atomic
			chunk_id++;

			int cur_sam_N;
			if(chunk_id == chunk_N) // the last round
				cur_sam_N = sam_N - (chunk_id - 1) * part_sam_N;
			else
				cur_sam_N = part_sam_N;



			checkCudaErrors(cudaMalloc(&d_sam_match[i],sizeof(int) * part_sam_N  * 5));
			checkCudaErrors(cudaMalloc(&d_sam_matchedCnt[i],sizeof(int) * part_sam_N));

			checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
			printf("Card %d after malloc %.2lf GB, total memory %.2lf GB\n",i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

			while(chunk_id < chunk_N)
				//the total number of sample points processed by this card
			{
#pragma omp atomic
				chunk_id++;

				int cur_sam_N;
				if(chunk_id == chunk_N) // the last round
					cur_sam_N = sam_N - (chunk_id - 1) * part_sam_N;
				else
					cur_sam_N = part_sam_N;

				int start_sam_pos = (chunk_id - 1) * part_sam_N;
				int end_sam_pos = start_sam_pos + cur_sam_N - 1;


				int start_pix = h_sam_pix[start_sam_pos];
				int end_pix = h_sam_pix[end_sam_pos];

				int start_ref_pos;
				if(start_pix == 0)
					start_ref_pos = 0;
				else
					start_ref_pos = binary_search(start_pix - 1,h_ref_pix,ref_N);
				//				start_ref_pos = get_start(start_pix,h_ref_node,ref_N);

				if(start_ref_pos == -1)
					continue;
				int end_ref_pos = binary_search(end_pix,h_ref_pix,ref_N) - 1;
				if(end_ref_pos == -2)
					end_ref_pos = ref_N - 1;
				int cur_ref_N = end_ref_pos - start_ref_pos + 1;

				dim3 block(block_size);
				dim3 grid(min(65536,(int)ceil(cur_sam_N * 1.0 / block.x)));

				if(cur_ref_N == 0)
					continue;

				printf("\n\nCard %d chunk-%d\n",i,chunk_id - 1);
				printf("block.x %d grid.x %d\n",block.x,grid.x);
				printf("start_sam_pos %d start_sam_pix %d end_sam_pos %d end_sam_pix %d sam_N %d\n",start_sam_pos,start_pix,end_sam_pos,end_pix,cur_sam_N);
				printf("start_ref_pos %d start_ref_pix %d end_ref_pos %d end_ref_pix %d ref_N %d\n",start_ref_pos,h_ref_pix[start_ref_pos],end_ref_pos,h_ref_pix[end_ref_pos],cur_ref_N);
				checkCudaErrors(cudaMemset(d_sam_matchedCnt[i],0,sizeof(int) * part_sam_N));
				checkCudaErrors(cudaMemcpy(d_sam_ra[i],h_sam_ra + start_sam_pos,cur_sam_N * sizeof(double),cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(d_sam_dec[i],h_sam_dec + start_sam_pos,cur_sam_N * sizeof(double),cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(d_sam_pix[i],h_sam_pix + start_sam_pos,cur_sam_N * sizeof(int),cudaMemcpyHostToDevice));

				checkCudaErrors(cudaMemcpy(d_ref_ra[i],h_ref_ra + start_ref_pos,cur_ref_N * sizeof(double), cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(d_ref_dec[i],h_ref_dec + start_ref_pos,cur_ref_N * sizeof(double), cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(d_ref_pix[i],h_ref_pix + start_ref_pos,cur_ref_N * sizeof(int), cudaMemcpyHostToDevice));

				kernel_singleCM<<<grid,block>>>(d_ref_ra[i],d_ref_dec[i],d_ref_pix[i],cur_ref_N,d_sam_ra[i],d_sam_dec[i],d_sam_pix[i],cur_sam_N,d_sam_match[i],d_sam_matchedCnt[i],start_ref_pos,start_sam_pos);

				checkCudaErrors(cudaMemcpy(h_sam_matchedCnt + start_sam_pos,d_sam_matchedCnt[i],cur_sam_N * sizeof(int),cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(h_sam_match + start_sam_pos * 5,d_sam_match[i],cur_sam_N * 5 * sizeof(int),cudaMemcpyDeviceToHost));
			}

			checkCudaErrors(cudaFree(d_sam_matchedCnt[i]));
			checkCudaErrors(cudaFree(d_sam_match[i]));
			checkCudaErrors(cudaFree(d_ref_ra[i]));
			checkCudaErrors(cudaFree(d_ref_dec[i]));
			checkCudaErrors(cudaFree(d_ref_pix[i]));
			checkCudaErrors(cudaFree(d_sam_ra[i]));
			checkCudaErrors(cudaFree(d_sam_dec[i]));
			checkCudaErrors(cudaFree(d_sam_pix[i]));

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
}
void worker_ownCM(int rank)
{
	worker_ref_N = pix_chunk_cnt[rank-1];
	int *ref_match = (int*)malloc(sizeof(int) * worker_ref_N * 5);
	int *ref_matchedCnt = (int*)malloc(sizeof(int) * worker_ref_N);

	printf("rank %d before tranfster to array\n",rank);
	struct timeval start,end;
	gettimeofday(&start,NULL);
	worker_toArray(pix_chunk_startPos[rank-1],pix_chunk_cnt[rank-1]);
	gettimeofday(&end,NULL);
	printf("rank-%d worker_toArray %.3f s\n",rank,diffTime(start,end) * 0.001);

	singleCM(h_worker_sam_ra,h_worker_sam_dec,h_worker_sam_pix,h_ref_dup_ra,h_ref_dup_dec,h_ref_dup_pix,worker_sam_N,worker_ref_N,ref_match,ref_matchedCnt);

	free(ref_match);
	free(ref_matchedCnt);
}
/*
   void worker_CM(int rank)
   {
   struct timeval start,end;
   MPI_Status status;
   MPI_Barrier(worker_comm);
   for(int i = 0; i < 1; ++i)
   {
   MPI_Barrier(worker_comm);
   int commID = request_node[rank-1][i]; // the rank which communiates to in this iteration
   int send_amount = pix_chunk_cnt[commID - 1];
   int startPos = pix_chunk_startPos[commID - 1];

   MPI_Sendrecv(&send_amount,1,MPI_INT,commID-1,3,&worker_ref_N,1,MPI_INT,commID-1,3,worker_comm,&status);
//		printf("rank-%d iteration-%d commID-%d will send %d recv %d\n",rank,i,commID,send_amount,worker_ref_N);
//		printf("rank-%d send startPos %d\n",rank,startPos);
h_worker_ref = (PIX_NODE *)malloc(sizeof(PIX_NODE) * worker_ref_N);
//		MPI_Sendrecv(h_ref_dup_node + pix_chunk_startPos[commID -1],send_amount,mpi_node,commID - 1,3,h_worker_ref,worker_ref_N,mpi_node,commID - 1,3,worker_comm,&status);

int min_amount = min(send_amount,worker_ref_N);
int sendRecv_iteration = ceil(1.0 * min_amount / MPI_MESSLEN);
int remain_amount = send_amount > worker_ref_N ? (send_amount - min_amount) : (worker_ref_N - min_amount);
int remain_iteration = ceil(1.0 * remain_amount / MPI_MESSLEN);

gettimeofday(&start,NULL);
for(int j = 0; j < sendRecv_iteration; ++j)
{
int cur_N;
if(j != sendRecv_iteration - 1)
cur_N = MPI_MESSLEN;
else
cur_N = min_amount - j * MPI_MESSLEN;
MPI_Sendrecv(h_ref_dup_node + startPos + j * MPI_MESSLEN,cur_N,mpi_node,commID-1,3,h_worker_ref + j * MPI_MESSLEN,cur_N,mpi_node,commID-1,3,worker_comm,&status);
}
gettimeofday(&end,NULL);
float tt = diffTime(start,end) * 0.001;
double throughPut = 2 * 20 * min_amount / 1024 / 1024 / 1024 / tt;
printf("rank-%d sendRecv %.3f s %.3lf GB/s\n",rank,diffTime(start,end) * 0.001,throughPut);


gettimeofday(&start,NULL);
for(int j = 0; remain_amount != 0 && j < remain_iteration; ++j)
{
int cur_N;
if(j != remain_iteration - 1)
cur_N = MPI_MESSLEN;
else
cur_N = remain_amount - j * MPI_MESSLEN;
if(send_amount > worker_ref_N)
{
MPI_Send(h_ref_dup_node + startPos +  min_amount + j * MPI_MESSLEN,cur_N,mpi_node,commID - 1,3,worker_comm);
//			printf("rank-%d send iteration-%d amount %d\n",rank,j,cur_N);
}
else
{
MPI_Recv(h_worker_ref + min_amount + j * MPI_MESSLEN,cur_N,mpi_node,commID - 1,3,worker_comm,&status);
//		MPI_Recv(buffer,cur_N,mpi_node,commID - 1,3,worker_comm,&status);
//		memcpy(h_worker_ref + min_amount + j * MPI_MESSLEN,buffer,cur_N * sizeof(PIX_NODE));
//			printf("rank-%d recv iteration-%d amount %d\n",rank,j,cur_N);
}
}
gettimeofday(&end,NULL);
printf("rank-%d send or recv %.3f s\n",rank,diffTime(start,end) * 0.001);

gettimeofday(&start,NULL);
int *ref_match = (int*)malloc(sizeof(int) * worker_ref_N * 5);
int *ref_matchedCnt = (int*)malloc(sizeof(int) * worker_ref_N);
singleCM(h_worker_sam,worker_sam_N,h_worker_ref,worker_ref_N,ref_match,ref_matchedCnt);
gettimeofday(&end,NULL);
printf("rank-%d single CM %.3f s\n",rank,diffTime(start,end) * 0.001);
free(ref_match);
free(ref_matchedCnt);
free(h_worker_ref);
}

}
/*
   void worker_CM_separateSendRecv(int rank)
   {
   struct timeval start,end;
//	buffer = (PIX_NODE*)malloc(sizeof(PIX_NODE) * MPI_MESSLEN);
MPI_Status status;
//	for(int i = 0; i < worker_N - 1; ++i)
for(int i = 0; i < 1; ++i)
{
int commID = request_node[rank-1][i]; // the rank which communiates to in this iteration
int send_amount = pix_chunk_cnt[commID - 1];
int startPos = pix_chunk_startPos[commID - 1];

MPI_Sendrecv(&send_amount,1,MPI_INT,commID-1,3,&worker_ref_N,1,MPI_INT,commID-1,3,worker_comm,&status);
printf("rank-%d iteration-%d commID-%d will send %d recv %d\n",rank,i,commID,send_amount,worker_ref_N);
printf("rank-%d send startPos %d\n",rank,startPos);
h_worker_ref = (PIX_NODE *)malloc(sizeof(PIX_NODE) * worker_ref_N);

int send_iteration = ceil(1.0 * send_amount / MPI_MESSLEN);
int recv_iteration = ceil(1.0 * worker_ref_N / MPI_MESSLEN);
int send_cnt = 0;
MPI_Request send_request[send_iteration];
MPI_Status send_status[send_iteration];

for(int j = 0; j < send_iteration; ++j)
{
int cur_N;
if(j != send_iteration - 1)
cur_N = MPI_MESSLEN;
else
cur_N = send_amount - j * MPI_MESSLEN;
MPI_Isend(h_ref_dup_node + startPos + j * MPI_MESSLEN,cur_N,mpi_node,commID-1,3,worker_comm,&send_request[send_cnt++]);
printf("rank-%d send iteration-%d amount %d\n",rank,j,cur_N);
}
for(int j = 0; j < recv_iteration; ++j)
{
int cur_N;
if(j != recv_iteration - 1)
cur_N = MPI_MESSLEN;
else
cur_N = worker_ref_N - j * MPI_MESSLEN;
MPI_Recv(h_worker_ref + j * MPI_MESSLEN,cur_N,mpi_node,commID-1,3,worker_comm,&status);
printf("rank-%d recv iteration-%d amount %d\n",rank,j,cur_N);
}

gettimeofday(&start,NULL);
int *ref_match = (int*)malloc(sizeof(int) * worker_ref_N * 5);
int *ref_matchedCnt = (int*)malloc(sizeof(int) * worker_ref_N);
singleCM(h_worker_sam,worker_sam_N,h_worker_ref,worker_ref_N,ref_match,ref_matchedCnt);
gettimeofday(&end,NULL);
printf("rank-%d single CM %.3f s\n",rank,diffTime(start,end) * 0.001);
free(ref_match);
free(ref_matchedCnt);
free(h_worker_ref);

MPI_Waitall(send_cnt,send_request,send_status);
}

}
*/
#endif
