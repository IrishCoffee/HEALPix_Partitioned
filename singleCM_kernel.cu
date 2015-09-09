#include "singleCM_kernel.h"
	
	__host__ __device__
int binary_search(int key, PIX_NODE *node, int N)
{
	int st = 0;
	int ed = N - 1;
	while(st < ed)
	{
		int mid = st + ((ed - st) >> 1);
		if(node[mid].pix <= key)
			st = mid + 1;
		else
			ed = mid;
	}
	if(node[ed].pix > key)
		return ed;
	return -1;
}
	__global__
void kernel_singleCM(PIX_NODE *ref_node, int ref_N, PIX_NODE *sam_node, int sam_N, int *sam_match,int *sam_matchedCnt,int ref_offset,int sam_offset)
{
	__shared__ int s_ref_pix[TILE_SIZE];
	__shared__ double s_ref_ra[TILE_SIZE];
	__shared__ double s_ref_dec[TILE_SIZE];

	__shared__ int start_pix,end_pix;
	__shared__ int start_ref_pos,end_ref_pos;
	__shared__ int block_sam_N,block_ref_N;
	__shared__ int iteration;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < sam_N)
		sam_matchedCnt[tid] = 0;


	if(threadIdx.x == 0)
	{
		if(blockIdx.x == gridDim.x - 1) // the last block
			block_sam_N = sam_N - blockIdx.x * blockDim.x;
		else
			block_sam_N = blockDim.x;

		start_pix = sam_node[tid].pix;
		end_pix = sam_node[tid + block_sam_N - 1].pix;

		if(start_pix == 0)
			start_ref_pos = 0;
		else
			start_ref_pos = binary_search(start_pix - 1,ref_node,ref_N);

		end_ref_pos = binary_search(end_pix,ref_node,ref_N);

		if(end_ref_pos == -1)
			end_ref_pos = ref_N - 1;
		else
			end_ref_pos--;
		block_ref_N = end_ref_pos - start_ref_pos + 1;
		iteration = ceil(block_ref_N * 1.0 / TILE_SIZE);
	}
	__syncthreads();
	if(start_ref_pos == -1 || end_ref_pos < start_ref_pos)
		return;

	int pix,cnt = 0;
	double sam_ra,sam_dec;
	if(tid < sam_N)
	{
		pix = sam_node[tid].pix;
		sam_ra = sam_node[tid].ra;
		sam_dec = sam_node[tid].dec;
		cnt = 0;
	}
	__syncthreads();
	for(int ite = 0; ite < iteration; ++ite)
	{
		__syncthreads();
		for(int k = 0; k < TILE_SIZE / blockDim.x; ++k)
		{
			int ref_pos = start_ref_pos + ite * TILE_SIZE + blockDim.x * k + threadIdx.x;
			int s_ref_pos = blockDim.x * k + threadIdx.x;
			if(ref_pos <= end_ref_pos)
			{
				s_ref_pix[s_ref_pos] = ref_node[ref_pos].pix;
				s_ref_ra[s_ref_pos] = ref_node[ref_pos].ra;
				s_ref_dec[s_ref_pos] = ref_node[ref_pos].dec;
			}
			else
				s_ref_pix[s_ref_pos] = -1;
		}
		__syncthreads();
		if(tid >= sam_N)
			continue;
		for(int j = 0; j < TILE_SIZE; ++j)
		{
			if(s_ref_pix[j] == -1 || s_ref_pix[j] > pix)
				break;
			if(s_ref_pix[j] < pix)
				continue;
			if(s_ref_pix[j] == pix && matched(sam_ra,sam_dec,s_ref_ra[j],s_ref_dec[j],0.0056))
			{
				cnt++;
				if(cnt <= 5)
					sam_match[tid * 5 + cnt - 1] = ref_offset + start_ref_pos + ite * TILE_SIZE + j;
			}
		}
	}

	if(tid < sam_N)
		sam_matchedCnt[tid] = cnt;
}


