#ifndef VALUES_H
#define VALUES_H

/* values used by worker nodes
*/
const double search_radius = 0.0056 * pi / 180.0;
cudaDeviceProp deviceProp;
const int worker_N = 6;

// for tesla K series cards gpu-10 to gpu-16 we have two GPU cards
const int GPU_N = 2;
const int GBSize = 1024 * 1024 * 1024;
const int cntSize = 805306368;
const int MPI_MESSLEN = 100000000;

char ref_table[240][64];
int ref_N,ref_dup_N;
int start_pix,end_pix;


int *h_R_cnt,*h_S_cnt;

double *h_ref_ra,*h_ref_dec;
int *h_ref_range;
int *h_ref_pix;

double *d_ref_ra[GPU_N],*d_ref_dec[GPU_N];
int *d_ref_range[GPU_N];

char sam_table[240][64];
int sam_N;
const int part_sam_N = 5000000;


//chunk information
int chunk_start_pix[worker_N];
int chunk_end_pix[worker_N];
int chunk_size[worker_N];


//values for MPI communication
int rank,numprocs,namelen;
char processor_name[MPI_MAX_PROCESSOR_NAME];
const int MASTER_NODE = 0;
MPI_Datatype mpi_node;
MPI_Group worker_group,entire_group;
MPI_Comm worker_comm,entire_comm;
MPI_Request send_request[6],recv_request;


//values for single_cm
int ref_CM_N,sam_CM_N;
PIX_NODE *h_ref_node, *h_sam_node;

int request_node[6][5] = {{2,3,4,5,6},{1,5,6,4,3},{4,1,5,6,2},{3,6,1,2,5},{6,2,3,1,4},{5,4,2,3,1}};
//int request_node[4][3] = {{2,3,4},{1,4,3},{4,1,2},{3,2,1}};
#endif

