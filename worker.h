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
	free(h_worker_ref);
	free(h_worker_sam);
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

void worker_merge(int rank)
{
	/*
	MPI_Status status;
	if(rank == 2 || rank == 3)
		MPI_Send(h_R_cnt,cntSize,MPI_INT,1,3,MPI_COMM_WORLD);
	else if(rank == 5 || rank == 6)
		MPI_Send(h_R_cnt,cntSize,MPI_INT,3,3,MPI_COMM_WORLD);
	else if(rank == 1)
	{
		int *cnt_buffer1 = (int*)malloc(sizeof(int) * cntSize);
		int *cnt_buffer2 = (int*)malloc(sizeof(int) * cntSize);
		int *cnt_merge = (int*)malloc(sizeof(int) * cntSize);
		MPI_Recv(cnt_buffer1,cntSize,MPI_INT,2,3,MPI_COMM_WORLD,&status);
		MPI_Recv(cnt_buffer2,cntSize,MPI_INT,3,3,MPI_COMM_WORLD,&status);
		for(int i = 0; i < cntSize; ++i)
			cnt_merge[i] = h_R_cnt[i] + cnt_buffer1[i] + cnt_buffer2[i];
		free(cnt_buffer1);
		free(cnt_buffer2);
		MPI_Send(cnt_merge,cntSize,MPI_INT,4,3,MPI_COMM_WORLD);
		free(cnt_merge);
	}
	else // rank 4
	{
		int *cnt_buffer1 = (int*)malloc(sizeof(int) * cntSize);
		int *cnt_buffer2 = (int*)malloc(sizeof(int) * cntSize);
		int *cnt_merge = (int*)malloc(sizeof(int) * cntSize);
		MPI_Recv(cnt_buffer1,cntSize,MPI_INT,5,3,MPI_COMM_WORLD,&status);
		MPI_Recv(cnt_buffer2,cntSize,MPI_INT,6,3,MPI_COMM_WORLD,&status);
		for(int i = 0; i < cntSize; ++i)
			cnt_merge[i] = h_R_cnt[i] + cnt_buffer1[i] + cnt_buffer2[i];
		free(cnt_buffer2);
		MPI_Recv(cnt_buffer1,cntSize,MPI_INT,1,3,MPI_COMM_WORLD,&status);
		long long sum = 0;
		for(int i = 0; i < cntSize; ++i)
		{
			cnt_merge[i] += cnt_buffer1[i];
			sum += cnt_merge[i];
		}
		free(cnt_buffer1);
		long long ave = sum / worker_N; // we have six nodes in total
		long long cur = 0;
		int start_pix = 0,end_pix = 0;;
		int rank_cnt = 0;
		for(int i = 0; i < cntSize; ++i)
		{
			if(cur + cnt_merge[i] > ave)
			{
				end_pix = i - 1;
				chunk_start_pix[rank_cnt] = start_pix;
				chunk_end_pix[rank_cnt] = end_pix;

				rank_cnt++;
				
				cur = cnt_merge[i];
				start_pix = i;
				
				if(rank_cnt == 5)
					break;
			}
			else
				cur += cnt_merge[i];
		}
		chunk_start_pix[rank_cnt] = start_pix;
		chunk_end_pix[rank_cnt] = cntSize - 1;
	}
	free(h_R_cnt);
	//rank 4 broadcast partition table 
	MPI_Barrier(worker_comm);
	MPI_Bcast(chunk_start_pix,6,MPI_INT,4,worker_comm);
	MPI_Bcast(chunk_end_pix,6,MPI_INT,4,worker_comm);
*/
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
/*
	omp_set_num_threads(worker_N);
#pragma omp parallel
	{
		int i = omp_get_thread_num() % worker_N;
		int cnt_tmp = 0;
		bool found_start = false;
		int start_pix = chunk_start_pix[i];
		int end_pix = chunk_end_pix[i];
		for(int j = 0; j < ref_dup_N; ++j)
		{
			if(h_ref_dup_node[j].pix >= start_pix && h_ref_dup_node[j].pix <= end_pix)
			{
				cnt_tmp++;
				if(!found_start)
				{
					found_start = true;
					pix_chunk_startPos[i] = j;
				}
			}
			if(h_ref_dup_node[j].pix > end_pix || j == ref_dup_N - 1)
			{
				pix_chunk_cnt[i] = cnt_tmp;
				break;
			}
			
		}
	}
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
	h_worker_sam = (PIX_NODE *)malloc(sizeof(PIX_NODE) * worker_sam_N);

	int ite = (int)ceil(worker_sam_N * 1.0 / MPI_MESSLEN);
	for(int i = 0; i < ite; ++i)
	{
		int len;
		if(i < ite - 1)
			len = MPI_MESSLEN;
		else
			len = worker_sam_N - i * MPI_MESSLEN;
		MPI_Recv(h_worker_sam + i * MPI_MESSLEN,len,mpi_node,MASTER_NODE,3,MPI_COMM_WORLD,&status);
		printf("rank-%d recv sample chunk %d\n",rank,i);
	}
}

void singleCM(PIX_NODE h_ref_node[], int ref_N, PIX_NODE h_sam_node[], int sam_N, int h_sam_match[],int h_sam_matchedCnt[])
{
	//the maximum number of sample points that can be matched each time by each card
	int part_sam_N = 25000000;
	int part_ref_N = 8 * part_sam_N;

	PIX_NODE *d_ref_node[GPU_N];
	PIX_NODE *d_sam_node[GPU_N];
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


		checkCudaErrors(cudaMalloc(&d_ref_node[i],sizeof(PIX_NODE) * part_ref_N));
		checkCudaErrors(cudaMalloc(&d_sam_node[i],sizeof(PIX_NODE) * part_sam_N));
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

			int start_pix = h_sam_node[start_sam_pos].pix;
			int end_pix = h_sam_node[end_sam_pos].pix;

			int start_ref_pos;
			if(start_pix == 0)
				start_ref_pos = 0;
			else
				start_ref_pos = binary_search(start_pix - 1,h_ref_node,ref_N);
			//				start_ref_pos = get_start(start_pix,h_ref_node,ref_N);

			if(start_ref_pos == -1)
				continue;
			int end_ref_pos = binary_search(end_pix,h_ref_node,ref_N) - 1;
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
			printf("start_ref_pos %d start_ref_pix %d end_ref_pos %d end_ref_pix %d ref_N %d\n",start_ref_pos,h_ref_node[start_ref_pos].pix,end_ref_pos,h_ref_node[end_ref_pos].pix,cur_ref_N);
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
void worker_ownCM(int rank)
{
	worker_ref_N = pix_chunk_cnt[rank-1];
	int *ref_match = (int*)malloc(sizeof(int) * worker_ref_N * 5);
	int *ref_matchedCnt = (int*)malloc(sizeof(int) * worker_ref_N);
	singleCM(h_worker_sam,worker_sam_N,h_ref_dup_node + pix_chunk_startPos[rank-1],worker_ref_N,ref_match,ref_matchedCnt);
	free(ref_match);
	free(ref_matchedCnt);
}
#endif
