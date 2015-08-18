#ifndef VALUES_H
#define VALUES_H

/* values used by worker nodes
*/
char ref_table[12][64];
const double search_radius = 0.0056 * pi / 180.0;
cudaDeviceProp deviceProp;


// for tesla K series cards gpu-10 to gpu-16 we have two GPU cards
const int GPU_N = 2;
const int GBSize = 1024 * 1024 * 1024;
const int cntSize = 805306368;

int ref_N,ref_dup_N;

double *h_ref_ra,*h_ref_dec;
int *h_ref_range;

int *h_ref_dup_pix,*h_ref_dup_pix1;
double *h_ref_dup_ra,*h_ref_dup_dec;

struct PIX_NODE
{
	double ra,dec;
	int pix;
};
PIX_NODE *h_ref_dup_node;

//count table information
int *h_R_cnt;
int *h_R_cnt_merged;
int *h_R_cnt_recv;
int *h_R_startPos;

double *d_ref_ra[GPU_N],*d_ref_dec[GPU_N];
int *d_ref_range[GPU_N];

/*values used by master node
 * */
char sam_table[240][64];
const int sam_N = 1200000000;
const int part_sam_N = 5000000;

PIX_NODE *h_sam_node;

//values for MPI communication
int rank,numprocs,namelen;
char processor_name[MPI_MAX_PROCESSOR_NAME];
const int MASTER_NODE = 0;
#endif

