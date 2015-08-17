#ifndef MASTER_H
#define MASTER_H

#include "values.h"
#include "helper_functions.h"

void master_allocation()
{
	h_sam_node = (PIX_NODE *)malloc(sizeof(PIX_NODE) * sam_N);
}

void master_free()
{
	free(h_sam_node);
}

//load sample file, compute HEALPix id for all points
void master_load_file(char *masterFileList)
{
	readSamFile(masterFileList,240);

	int offset = 5000000;
	int eachN = 240 / 40; //each thread is responsible for 6 files

	omp_set_num_threads(40);
#pragma omp parallel
	{
		int threadId = omp_get_thread_num() % 40;
		for(int i = threadId * eachN; i < (threadId + 1) * eachN; ++i)
			readSampleData(sam_table[i],h_sam_node + i * offset,offset);
	}
}

void master_getPix()
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


#endif
