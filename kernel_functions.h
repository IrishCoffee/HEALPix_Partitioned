#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H

#include "healpix_base.h"
#include "STL_CUDA.h"

__global__ void getPix(double *d_ra,double *d_dec,int *d_pix,int N);
__global__ void get_PixRange(double *d_ra,double *d_dec,int *d_range,double radius, int N);

//__global__ void query_disc(int *d_id,int *d_pix,double *d_ra,double *d_dec,int *d_range,double radius,int N);
__global__ void query_disc(double *ref_ra,double *ref_dec,int *sam_pix,double * sam_ra,double *sam_dec,int * ref_cnt,int *ref_range,int *d_result,double radius,int N_ref,int N_sam,int offset_result);

//__device__ void query_disc_internal(Point ptg, double radius, int fact,int order, Pair_Vector &pixset);
__device__ void query_disc_internal(Point ptg, double radius, int fact,int order, Pair_Vector &pixset,Healpix_Base base[],double crpdr[],double crmd[]);

__device__ void check_pixel(int o,int order_,int omax, int zone, Pair_Vector &pixset, int pix, Stack &stk, bool inclusive, int &stacktop);

__device__ int begin_index(int id,int *d_pix,int N);

#endif
