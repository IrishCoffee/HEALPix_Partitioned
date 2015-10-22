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

	h_sam_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * sam_N);
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
		if(readSampleData(sam_table[rank * load_file_num  + i],h_sam_node + i * offset,offset,sam_ignore) == -1)
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
		double z = cos(radians(h_sam_node[i].dec + 90.0));
		double phi = radians(h_sam_node[i].ra);
		h_sam_node[i].pix = healpix_test.zphi2pix(z,phi);
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

void cal_refChunk_size(int rank)
{
	memset(chunk_size,0,sizeof(chunk_size));
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
						chunk_size[chunk_id]++;
			}
		}
	}
	for(int i = 0; i < worker_N; ++i)
		printf("rank-%d chunk-%d size %d\n",rank,i,chunk_size[i]);
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
	memset(chunk_size,0,sizeof(chunk_size));
#pragma omp parallel for
	for(int i = 0; i < worker_N; ++i)
	{
		int lower = chunk_start_pix[i];
		int upper = chunk_end_pix[i];
		for(int j = lower; j <= upper; ++j)
			chunk_size[i] += h_R_cnt[j];
	}
	for(int i = 0; i < worker_N; ++i)
		printf("rank-%d chunk-%d size-%d\n",rank,i,chunk_size[i]);
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

void receive_R_chunk(int source,PIX_NODE *ref_node,int size)
{
	MPI_Status status;
	int ite = (size - 1) / MPI_MESSLEN;
	for(int i = 0; i < ite; ++i)
	{
		int recv_N;
		if(i == ite - 1)
			recv_N = size - i * MPI_MESSLEN;
		else
			recv_N = MPI_MESSLEN;

		MPI_Recv(ref_node,recv_N,mpi_node,source,3,MPI_COMM_WORLD,&status);
	}
}

void redistribute_R(int rank)
{
	MPI_Status status;
	int each_chunk_N[worker_N];
	each_chunk_N[rank] = chunk_size[rank];

	PIX_NODE *ref_buffer,*more_ref_buffer;
	omp_set_num_threads(worker_N);
//#pragma omp parallel
	for(int i = 0; i < worker_N; ++i)
	{
//		int i = omp_get_thread_num() % worker_N;
		if(i == rank)
		{
			/*
			for(int j = 0; j < worker_N; ++j)
			{
				if(j != rank)
					MPI_Recv(each_chunk_N + j,1,MPI_INT,j,3,MPI_COMM_WORLD,&status);
			}
			ref_CM_N = 0;
			for(int j = 0; j < worker_N; ++j)
				ref_CM_N += each_chunk_N[j];
			h_ref_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * ref_CM_N);

			/*
			int offset = 0;
			for(int j = 0; j < worker_N; ++j)
			{
				if(j == rank)
					continue;
				receive_R_chunk(j,h_ref_node + offset,each_chunk_N[j]);
				offset += each_chunk_N[j];
			}
			*/
		}
		else
		{
	//		MPI_Send(chunk_size + i,1,MPI_INT,i,3,MPI_COMM_WORLD);

			more_ref_buffer = (PIX_NODE *)realloc(ref_buffer,sizeof(PIX_NODE) * chunk_size[i]);;
			ref_buffer = more_ref_buffer;
			struct timeval start,end;
			gettimeofday(&start,NULL);
			transfer_ref(rank,i,ref_buffer,chunk_size[i]);
			gettimeofday(&end,NULL);
			printf("rank-%d transfer chunk-%d costs %.3f \n",rank,i,diffTime(start,end) * 0.001);
		}
	}
	free(ref_buffer);
	free(more_ref_buffer);
	/*
	cout << "\\\\\n" << endl;
	for(int i = 0; i < worker_N; ++i)
		cout << "rank-" << rank << " chunk-" << i << " size " << each_chunk_N[i] << endl;
		*/
}

#endif
