#ifndef VALUES_H
#define VALUES_H

/* values used by worker nodes
*/
char ref_table[240][64];
const double search_radius = 0.0056 * pi / 180.0;
cudaDeviceProp deviceProp;


// for tesla K series cards gpu-10 to gpu-16 we have two GPU cards
const int GPU_N = 2;
const int GBSize = 1024 * 1024 * 1024;
const int cntSize = 805306368;
const int worker_N = 6;
const int MPI_MESSLEN = 100000000;
//const int MPI_MESSLEN = 209715;

int ref_N,ref_dup_N;
int start_pix,end_pix;

double *h_ref_ra,*h_ref_dec;
int *h_ref_range;
int *h_ref_pix;

int *h_ref_dup_pix;
double *h_ref_dup_ra,*h_ref_dup_dec;

PIX_NODE *h_ref_dup_node;

int worker_sam_N;
int worker_ref_N;
double *h_worker_sam_ra;
double *h_worker_sam_dec;
int *h_worker_sam_pix;
double *h_worker_ref_ra;
double *h_worker_ref_dec;
int *h_worker_ref_pix;


int pix_chunk_startPos[6];
int pix_chunk_cnt[6];
int chunk_start_pix[6];
int chunk_end_pix[6];



//count table information
int *h_R_cnt;
int *h_R_cnt_merged;

double *d_ref_ra[GPU_N],*d_ref_dec[GPU_N];
int *d_ref_range[GPU_N];

/*values used by master node
 * */
char sam_table[240][64];
const int sam_N = 1200000000;
const int part_sam_N = 5000000;

PIX_NODE *h_sam_node;
double *h_sam_ra;
double *h_sam_dec;
int *h_sam_pix;

//values for MPI communication
int rank,numprocs,namelen;
char processor_name[MPI_MAX_PROCESSOR_NAME];
const int MASTER_NODE = 0;
MPI_Datatype mpi_node;
MPI_Group worker_group,entire_group;
MPI_Comm worker_comm,entire_comm;
MPI_Request send_request[6],recv_request;


int request_node[6][5] = {{2,3,4,5,6},{1,5,6,4,3},{4,1,5,6,2},{3,6,1,2,5},{6,2,3,1,4},{5,4,2,3,1}};
//int request_node[4][3] = {{2,3,4},{1,4,3},{4,1,2},{3,2,1}};
#endif

