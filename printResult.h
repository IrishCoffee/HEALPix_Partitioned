#include <iostream>
#include <cstdio>
#include "constants.h"
using namespace std;

//print the pix id for sample point
void print_pix(int h_pix[],int h_id[],double h_ra[],double h_dec[],int N)
{
	freopen("out_Pix","w",stdout);
	for(int i = 0; i < N; ++i)
		printf("%d %d %lf %lf\n",h_id[i],h_pix[i],h_ra[i],h_dec[i]);
	fclose(stdout);
	freopen("/dev/tty","w",stdout);
}
void print_matchedResult(int h_ref_id[],double h_ref_ra[],double h_ref_dec[],double h_sam_ra[],double h_sam_dec[],int h_sam_id[],int h_ref_cnt[],int h_ref_result[],int N)
{
	int matchedCount = 0;
	freopen("out_matchedResult","w",stdout);
	for(int i = 0; i < N; ++i)
	{
		printf("\n------------------------\n%d %lf %lf\n=================================\n",h_ref_id[i],h_ref_ra[i],h_ref_dec[i]);
		for(int j = i * MAX_MATCH; j < MAX_MATCH * i + h_ref_cnt[i]; ++j)
		{
			printf("%d %lf %lf %lf\n",h_sam_id[h_ref_result[j]],h_sam_ra[h_ref_result[j]],h_sam_dec[h_ref_result[j]],0.0);
			matchedCount++;
		}
	}
	/*
	   for(int i = 0; i < N; ++i)
	   {
	   for(int j = i * MAX_MATCH; j < MAX_MATCH * i + h_cnt[i]; ++j)
	   {
	   printf("%d %lf %lf ",h_id[i],h_ra[i],h_dec[i]);
	   printf("%d %lf %lf %lf\n",h_id[h_result[j]],h_ra[h_result[j]],h_dec[h_result[j]],0.0);
	   }
	   }
	 */
	printf("Matched %d\n",matchedCount);
	fclose(stdout);
	freopen("/dev/tty","w",stdout);
}
void print_matchedCount(int h_id[],double h_ra[],double h_dec[],int h_cnt[],int h_result[],int N)
{
	freopen("out_matchedCount","w",stdout);
	int sum = 0;
	for(int i = 0; i < N; ++i)
	{
		printf("%d %d\n",h_id[i],h_cnt[i]);
		sum += h_cnt[i];
	}
	fclose(stdout);
	freopen("/dev/tty","w",stdout);
	printf("sum %d\n",sum);
}
int get_matchedCount(int h_cnt[],int N)
{
	int sum = 0;
	for(int i = 0; i < N; ++i)
		sum += h_cnt[i];
	return sum;
}
