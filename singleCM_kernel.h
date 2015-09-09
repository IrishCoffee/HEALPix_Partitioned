#ifndef SINGLECM_KERNEL_H
#define SINGLECM_KERNEL_H

#include "geometry_cal.h"
#include "constants.h"

	__host__ __device__
int binary_search(int key, PIX_NODE *node, int N);
	__global__
void kernel_singleCM(PIX_NODE *ref_node, int ref_N, PIX_NODE *sam_node, int sam_N, int *sam_match,int *sam_matchedCnt,int ref_offset,int sam_offset);

#endif
