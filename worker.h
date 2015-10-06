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
	int offset = ref_N / 2 / 20;

#pragma omp parallel for
	for(int i = 0; i < 20; ++i)
	{
		if(readDataFile(ref_table[workerID * 40  + i],h_ref_ra + i * offset,h_ref_dec + i * offset,offset) == -1)
			printf("load file %s error!\n",ref_table[workerID * 40 + i]);
	}
#pragma omp parallel for
	for(int i = 20; i < 40; ++i)
	{
		if(readDataFile(ref_table[workerID * 40  + i],h_ref_ra + i * offset,h_ref_dec + i * offset,offset) == -1)
			printf("load file %s error!\n",ref_table[workerID * 40 + i]);
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

void worker_duplicateR()
{
	h_ref_dup_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * ref_N * 8);

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
void worker_gather(int rank)
{
	MPI_Barrier(worker_comm);
	int worker_root = 1;
	MPI_Status status;
	if(rank == worker_root)
	{
		int *cnt_buffer = (int*)malloc(sizeof(int) * cntSize);
		int ite = (cntSize-1) / MPI_MESSLEN + 1;
		printf("rank-%d  before gather\n",rank);
		for(int j = 0; j < worker_N - 1; ++j)
		{
			MPI_Recv(cnt_buffer,cntSize,MPI_INT,j+2,3,MPI_COMM_WORLD,&status);
			printf("rank-%d recv from rank-%d cntChunk\n",rank,j+2);
#pragma omp parallel for
			for(int i = 0; i < cntSize; ++i)
				h_R_cnt[i] += cnt_buffer[i];
		}
//		MPI_Gather(NULL,0,MPI_INT,cnt_buffer,cntSize,MPI_INT,worker_root,worker_comm);
		printf("rank-%d  after gather\n",rank);
		free(cnt_buffer);

		long long sum = 0;
		for(int i = 0; i < cntSize; ++i)
			sum += h_R_cnt[i];
		double avg_sum = sum * 1.0 / worker_N;
		int cur_sum = 0;
		int prev_sid = 0;
		int cal_cnt = 0;
		cout << "avg sum " << avg_sum << endl;
		for(int i = 0; i < cntSize; ++i)
		{
			if(cur_sum + h_R_cnt[i] <= avg_sum)
				cur_sum += h_R_cnt[i];
			else
			{
				chunk_start_pix[cal_cnt] = prev_sid;
				chunk_end_pix[cal_cnt] = i - 1;
				printf("[%d,%d] -> %d\n",prev_sid,i - 1,cur_sum);
				cal_cnt++;
				prev_sid = i;
				cur_sum = h_R_cnt[i];
				if(cal_cnt == worker_N - 1)
					break;
			}
		}
		chunk_start_pix[cal_cnt] = prev_sid;
		chunk_end_pix[cal_cnt] = cntSize - 1;
		printf("[%d,%d]\n",prev_sid,cntSize - 1);
		MPI_Bcast(chunk_start_pix,worker_N,MPI_INT,worker_root-1,worker_comm);
		MPI_Bcast(chunk_end_pix,worker_N,MPI_INT,worker_root - 1,worker_comm);
	}
	else
	{
		printf("rank-%d  before gather\n",rank);
		MPI_Send(h_R_cnt,cntSize,MPI_INT,worker_root,3,MPI_COMM_WORLD);
		printf("rank-%d  after gather\n",rank);
		MPI_Bcast(chunk_start_pix,worker_N,MPI_INT,worker_root-1,worker_comm);
		MPI_Bcast(chunk_end_pix,worker_N,MPI_INT,worker_root - 1,worker_comm);
	}
	printf("rank-%d finished broadcast\n",rank);
}
void worker_merge(int rank)
{
	/*
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
	*/
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

void singleCM(int rank,double *h_ref_ra,double *h_ref_dec,int *h_ref_pix,double *h_sam_ra,double *h_sam_dec,int *h_sam_pix,int ref_N,int sam_N,int *h_sam_match,int *h_sam_matchedCnt)
{
	//the maximum number of sample points that can be matched each time by each card
	int part_sam_N = 35000000;
	int part_ref_N = 4 * part_sam_N;

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
		//	printf("rank-%d Card %d before malloc %.2lf GB, total memory %.2lf GB\n",rank,i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

		checkCudaErrors(cudaMalloc(&d_ref_ra[i],sizeof(double) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_dec[i],sizeof(double) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_ref_pix[i],sizeof(int) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_sam_ra[i],sizeof(double) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_dec[i],sizeof(double) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_pix[i],sizeof(int) * part_sam_N));
		checkCudaErrors(cudaMalloc(&d_sam_match[i],sizeof(int) * part_sam_N  * 5));
		checkCudaErrors(cudaMalloc(&d_sam_matchedCnt[i],sizeof(int) * part_sam_N));

		checkCudaErrors(cudaMemGetInfo(&free_mem,&total_mem));
		//	printf("rank-%d Card %d after malloc %.2lf GB, total memory %.2lf GB\n",rank,i,free_mem * 1.0 / GBSize,total_mem * 1.0 / GBSize);

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

			//		printf("\n\nrank-%d Card %d chunk-%d\n",rank,i,chunk_id - 1);
			//		printf("rank-%d block.x %d grid.x %d\n",rank,block.x,grid.x);
			//		printf("rank-%d start_sam_pos %d start_sam_pix %d end_sam_pos %d end_sam_pix %d sam_N %d\n",rank,start_sam_pos,start_pix,end_sam_pos,end_pix,cur_sam_N);
			//		printf("rank-%d start_ref_pos %d start_ref_pix %d end_ref_pos %d end_ref_pix %d ref_N %d\n",rank,start_ref_pos,h_ref_pix[start_ref_pos],end_ref_pos,h_ref_pix[end_ref_pos],cur_ref_N);
			checkCudaErrors(cudaMemset(d_sam_matchedCnt[i],0,sizeof(int) * part_sam_N));
			checkCudaErrors(cudaMemcpy(d_sam_ra[i],h_sam_ra + start_sam_pos,cur_sam_N * sizeof(double),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_sam_dec[i],h_sam_dec + start_sam_pos,cur_sam_N * sizeof(double),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_sam_pix[i],h_sam_pix + start_sam_pos,cur_sam_N * sizeof(int),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_ra[i],h_ref_ra + start_ref_pos,cur_ref_N * sizeof(double),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_dec[i],h_ref_dec + start_ref_pos,cur_ref_N * sizeof(double),cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_ref_pix[i],h_ref_pix + start_ref_pos,cur_ref_N * sizeof(int),cudaMemcpyHostToDevice));
			kernel_singleCM<<<grid,block>>>(d_ref_ra[i],d_ref_dec[i],d_ref_pix[i],cur_ref_N,d_sam_ra[i],d_sam_dec[i],d_sam_pix[i],cur_sam_N,d_sam_match[i],d_sam_matchedCnt[i],start_ref_pos,start_sam_pos);

			checkCudaErrors(cudaMemcpy(h_sam_matchedCnt + start_sam_pos,d_sam_matchedCnt[i],cur_sam_N * sizeof(int),cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(h_sam_match + start_sam_pos * 5,d_sam_match[i],cur_sam_N * 5 * sizeof(int),cudaMemcpyDeviceToHost));
		}

		checkCudaErrors(cudaFree(d_sam_matchedCnt[i]));
		checkCudaErrors(cudaFree(d_sam_match[i]));
		checkCudaErrors(cudaFree(d_ref_pix[i]));
		checkCudaErrors(cudaFree(d_ref_ra[i]));
		checkCudaErrors(cudaFree(d_ref_dec[i]));
		checkCudaErrors(cudaFree(d_sam_pix[i]));
		checkCudaErrors(cudaFree(d_sam_ra[i]));
		checkCudaErrors(cudaFree(d_sam_dec[i]));
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
	cout << "rank-" << rank << " sum " << sum << endl;
	cout << "rank-" << rank << " ave " << sum * 1.0 / sam_N << endl;
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
	singleCM(rank,h_worker_sam_ra,h_worker_sam_dec,h_worker_sam_pix,h_ref_dup_ra,h_ref_dup_dec,h_ref_dup_pix,worker_sam_N,worker_ref_N,ref_match,ref_matchedCnt);

	free(ref_match);
	free(ref_matchedCnt);
}
void worker_CM(int rank)
{
	int send_cnt = 0;
	struct timeval ite_start,ite_end;
	struct timeval start,end;
	//	buffer = (PIX_NODE*)malloc(sizeof(PIX_NODE) * MPI_MESSLEN);
	MPI_Status status;
	for(int i = 0; i < worker_N - 1; ++i)
	{
		gettimeofday(&ite_start,NULL);
//		MPI_Barrier(worker_comm);
		printf("\n\nrank-%d ITERATION %d\n",rank,i);
		int commID = request_node[rank-1][i]; // the rank which communiates to in this iteration
		int send_amount = pix_chunk_cnt[commID - 1];
		int startPos = pix_chunk_startPos[commID - 1];

		MPI_Sendrecv(&send_amount,1,MPI_INT,commID-1,3,&worker_ref_N,1,MPI_INT,commID-1,3,worker_comm,&status);
		printf("rank-%d iteration-%d commID-%d will send %d recv %d\n",rank,i,commID,send_amount,worker_ref_N);
		printf("rank-%d send startPos %d\n",rank,startPos);
		h_worker_ref_ra = (double *)malloc(sizeof(double) * worker_ref_N);
		h_worker_ref_dec = (double *)malloc(sizeof(double) * worker_ref_N);
		h_worker_ref_pix = (int *)malloc(sizeof(int) * worker_ref_N);


		int send_iteration = ceil(1.0 * send_amount / MPI_MESSLEN);
		int recv_iteration = ceil(1.0 * worker_ref_N / MPI_MESSLEN);
		MPI_Status status;

		struct timeval ss1,ee1;
		gettimeofday(&ss1,NULL);
		for(int kk = 0; kk < 2; ++kk)
		{
			if((kk == 0 && commID > rank) || (kk == 1 && commID < rank))
			{
				for(int j = 0; j < send_iteration; ++j)
				{
					int cur_N;
					if(j != send_iteration - 1)
						cur_N = MPI_MESSLEN;
					else
						cur_N = send_amount - j * MPI_MESSLEN;
					worker_toArray(startPos + j * MPI_MESSLEN,cur_N);
		//			   MPI_Isend(h_ref_dup_ra,cur_N,MPI_DOUBLE,commID-1,3,worker_comm,&send_request[send_cnt++]);
		//			   MPI_Isend(h_ref_dup_dec,cur_N,MPI_DOUBLE,commID-1,3,worker_comm,&send_request[send_cnt++]);
		//			   MPI_Isend(h_ref_dup_pix,cur_N,MPI_INT,commID-1,3,worker_comm,&send_request[send_cnt++]);
					MPI_Send(h_ref_dup_ra,cur_N,MPI_DOUBLE,commID-1,3,worker_comm);
					MPI_Send(h_ref_dup_dec,cur_N,MPI_DOUBLE,commID-1,3,worker_comm);
					MPI_Send(h_ref_dup_pix,cur_N,MPI_INT,commID-1,3,worker_comm);
					printf("rank-%d send to rank-%d iteration-%d amount %d\n",rank,commID,j,cur_N);
				}
			}
			else if((kk == 0 && commID < rank) || (kk == 1 && commID > rank))
			{
				for(int j = 0; j < recv_iteration; ++j)
				{
					int cur_N;
					if(j != recv_iteration - 1)
						cur_N = MPI_MESSLEN;
					else
						cur_N = worker_ref_N - j * MPI_MESSLEN;
					/*
					   MPI_Irecv(h_worker_ref_ra + j * MPI_MESSLEN,cur_N,MPI_DOUBLE,commID - 1,3,worker_comm,&recv_request[recv_cnt++]);
					   MPI_Irecv(h_worker_ref_dec + j * MPI_MESSLEN,cur_N,MPI_DOUBLE,commID - 1,3,worker_comm,&recv_request[recv_cnt++]);
					   MPI_Irecv(h_worker_ref_pix + j * MPI_MESSLEN,cur_N,MPI_INT,commID - 1,3,worker_comm,&recv_request[recv_cnt++]);
					   */
					MPI_Recv(h_worker_ref_ra + j * MPI_MESSLEN,cur_N,MPI_DOUBLE,commID - 1,3,worker_comm,&status);
					MPI_Recv(h_worker_ref_dec + j * MPI_MESSLEN,cur_N,MPI_DOUBLE,commID - 1,3,worker_comm,&status);
					MPI_Recv(h_worker_ref_pix + j * MPI_MESSLEN,cur_N,MPI_INT,commID - 1,3,worker_comm,&status);
					printf("rank-%d recv from rank-%d iteration-%d amount %d\n",rank,commID,j,cur_N);
				}
			}
		}
		gettimeofday(&ee1,NULL);
		printf("rank-%d communication in iteartion-%d cost %.3f s\n",rank,i,diffTime(ss1,ee1) * 0.001);

		gettimeofday(&start,NULL);
		int *ref_match = (int*)malloc(sizeof(int) * worker_ref_N * 5);
		int *ref_matchedCnt = (int*)malloc(sizeof(int) * worker_ref_N);
		singleCM(rank,h_worker_sam_ra,h_worker_sam_dec,h_worker_sam_pix,h_worker_ref_ra,h_worker_ref_dec,h_worker_ref_pix,worker_sam_N,worker_ref_N,ref_match,ref_matchedCnt);
		gettimeofday(&end,NULL);
		printf("rank-%d iteration %d single CM %.3f s\n",rank,i,diffTime(start,end) * 0.001);
		free(ref_match);
		free(ref_matchedCnt);
		free(h_worker_ref_ra);
		free(h_worker_ref_dec);
		free(h_worker_ref_pix);
		gettimeofday(&ite_end,NULL);
		printf("rank-%d iteration %d cost %.3f s\n",rank,i,diffTime(ite_start,ite_end) * 0.001);
	}

}
#endif
