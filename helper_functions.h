#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include "values.h"
double diffTime(timeval start,timeval end)
{
	return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

void readDataFile(char *fileName,double ra[],double dec[],int N)
{
	FILE * fd = fopen(fileName,"r");
	for(int i = 0; i < N; ++i)
		fscanf(fd,"%lf%lf%*d",&ra[i],&dec[i]);
	fclose(fd);
}
void readRefFile(char * tableList,int N)
{
	FILE *fd = fopen(tableList,"r");
	for(int i = 0; i < N; ++i)
		fscanf(fd,"%s",ref_table[i]);
	fclose(fd);
}

#endif
