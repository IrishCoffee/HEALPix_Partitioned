#ifndef SINGLECM_KERNEL_H
#define SINGLECM_KERNEL_H

#include "geometry_cal.h"
#include "constants.h"

	__host__ __device__
int binary_search(int key, int *pix, int N);
	__host__ __device__
int binary_search(int key, PIX_NODE *node, int N);

__global__
void kernel_singleCM(double *ref_ra,double *ref_dec,int *ref_pix,int ref_N,double *sam_ra,double *sam_dec,int *sam_pix,int sam_N,int *sam_match,int *sam_matchedCnt,int ref_offset,int sam_offset);

#endif
