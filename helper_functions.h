#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include "values.h"
double diffTime(timeval start,timeval end)
{
	return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

int readDataFile(char *fileName,double ra[],double dec[],int N,int ignore)
{
	FILE * fd = fopen(fileName,"r");
	if(fd == NULL)
		return -1;
	if(ignore)
	{
		for(int i = 0; i < N; ++i)
			fscanf(fd,"%lf%lf%*d",&ra[i],&dec[i]);
	}
	else
	{
		for(int i = 0; i < N; ++i)
			fscanf(fd,"%lf%lf",&ra[i],&dec[i]);
	}
	fclose(fd);
	return 0;
}

int readSampleData(char *fileName,PIX_NODE sam_node[],int N,int ignore)
{
	FILE * fd = fopen(fileName,"r");
	if(fd == NULL)
		return -1;
	if(ignore)
	{
		for(int i = 0; i < N; ++i)
			fscanf(fd,"%lf%lf%*d",&sam_node[i].ra,&sam_node[i].dec);
	}
	else
	{
		for(int i = 0; i < N; ++i)
			fscanf(fd,"%lf%lf",&sam_node[i].ra,&sam_node[i].dec);
	}
	fclose(fd);
	return 0;
}

int readRefFile(char * tableList,int N)
{
	FILE *fd = fopen(tableList,"r");
	if(fd == NULL)
		return -1;
	for(int i = 0; i < N; ++i)
		fscanf(fd,"%s",ref_table[i]);
	fclose(fd);
	return 0;
}

int readSamFile(char *tableList,int N)
{
	FILE *fd = fopen(tableList,"r");
	if(fd == NULL)
		return -1;
	for(int i = 0; i < N; ++i)
		fscanf(fd,"%s",sam_table[i]);
	fclose(fd);
	return 0;
}
bool cmp(PIX_NODE node_a,PIX_NODE node_b)
{
	return node_a.pix < node_b.pix;
}


#endif
