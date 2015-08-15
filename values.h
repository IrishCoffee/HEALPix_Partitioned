#ifndef VALUES_H
#define VALUES_H

/* values used by worker nodes
*/
char ref_table[12][64];
char sam_table[12][64];

// for tesla K series cards gpu-10 to gpu-16 we have two GPU cards
const int GPU_N = 2;
const int GBSize = 1024 * 1024 * 1024;

int ref_N,sam_N,ref_dup_N;

double *h_ref_ra,*h_ref_dec;
int *h_ref_range;

struct REF_NODE
{
	double ra,dec;
	int pix;
};
REF_NODE *ref_dup_node;

double *d_ref_ra[GPU_N],*d_ref_dec[GPU_N];
int *d_ref_range[GPU_N];

#endif

