#include "kernel_functions.h"

	__global__ 
void getPix(double *d_ra, double *d_dec, int *d_pix,int N)
{
	__shared__ Healpix_Base healpixTest;
	healpixTest.set_order(max_order);
	__syncthreads();

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNumber = blockDim.x * gridDim.x;

	if(threadId >= N)
		return;

	double z,phi;
	while(threadId < N)
	{
		z = cos(radians(d_dec[threadId] + 90));
		phi = radians(d_ra[threadId]);
		d_pix[threadId] = healpixTest.zphi2pix(z,phi);
		threadId += threadNumber;
	}
}

//__global__ void query_disc(int *id,int *pix, double *ra,double *dec,int *range,double radius, int N)
	__global__ 
void get_PixRange(double *ref_ra,double *ref_dec,int *range,double radius, int N)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNumber = blockDim.x * gridDim.x;
	
	__shared__ Healpix_Base base[max_order + 1];
	__shared__ double crpdr[max_order + 1];
	__shared__ double crmdr[max_order + 1];
	if(threadIdx.x <= max_order)
    {
		int i = threadIdx.x;
		base[i].set_order(i);
		double dr = base[i].max_pixrad();
		crpdr[i] = (radius + dr > pi) ? -1.0 : cos(radius + dr);
		crmdr[i] = (radius - dr < 0.0) ? 1.0 : cos(radius - dr);
	}
	__syncthreads();

	Pair_Vector pixset;
	double z;
	double phi;
	while(threadId < N)
	{
		pixset.clear();
		z = cos(radians(ref_dec[threadId] + 90.0));
		phi = radians(ref_ra[threadId]);

		Point ptg(z,phi);
		query_disc_internal(ptg,2 * radius, 0,13,pixset,base,crpdr,crmdr); 

		//assume there are at most 10 range pairs
		int pos = threadId * 20;

		while(!pixset.isEmpty())
		{
			range[pos] = pixset.back().first;
			range[pos+1] = pixset.back().second;
			
			pixset.pop_back();
			pos += 2;
		}
		range[pos] = -1;
		threadId += threadNumber;
	}
	return;
}
	__global__ 
void query_disc(double *ref_ra,double *ref_dec,int *sam_pix,double *sam_ra,double *sam_dec,int *cnt,int *range,int *result,double radius,int N_ref,int N_sam,int offset_result)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNumber = blockDim.x * gridDim.x;
	bool flag = false;

	while(threadId < N_ref)
	{

		int matchedCnt = cnt[threadId];
		int lower = 0;
		int upper = N_sam;
		int begin_pos;
		int pos = threadId * MAX_MATCH + cnt[threadId];
		double refRa = ref_ra[threadId];
		double refDec = ref_dec[threadId];

		for(int j = threadId * 20; range[j] != -1; j += 2)
		{
			lower = range[j];
			upper = range[j+1];
			begin_pos = begin_index(lower,sam_pix,N_sam);
			if(begin_pos == -1)
				continue;

			for(int i = begin_pos; i < N_sam && sam_pix[i] <= upper; ++i)
			{
				if(matched(sam_ra[i],sam_dec[i],refRa,refDec,radius))
				{
					matchedCnt++;
					if(matchedCnt <= MAX_MATCH) // only record the previous 64 records
						result[pos++] = offset_result + i;
				}
				/*
				if(matchedCnt == MAX_MATCH)
					break;
				 */
			}
		}
		cnt[threadId] = matchedCnt;
		threadId += threadNumber;
	}
	return;
}

	__device__ 
void query_disc_internal(Point ptg, double radius, int fact,int order, Pair_Vector &pixset,Healpix_Base base[],double crpdr[],double crmdr[])
{
	bool inclusive = (fact != 0);
	int oplus = 0;
	int omax = order + oplus;
	Vec3 vptg(ptg.z,ptg.phi);
	double cosrad = cos(radius);


	Stack stk;
	for(int i = 0; i < 12; ++i)
	{
		Pair tmp(11-i,0);
		stk.push_back(tmp);
	}

	int stacktop = 0;
	Pair top = stk.back();
	while(!stk.isEmpty())
	{
		int pix = stk.back().first;
		int o = stk.back().second;
		stk.pop_back();

		double z,phi;
		base[o].pix2zphi(pix,z,phi);

		//cosine of angular distance between pixel center and disk center
		double cangdist = cosdist_zphi(vptg.z, ptg.phi,z,phi);

		if(cangdist > crpdr[o])
		{
			int zone = (cangdist < cosrad) ? 1 : ((cangdist <= crmdr[o]) ? 2 : 3);
			check_pixel(o,order,omax,zone,pixset,pix,stk,inclusive,stacktop);
		}
	}
	return;
}

	__device__ 
void check_pixel(int o, int order,int omax, int zone, Pair_Vector &pixset, int pix, Stack &stk, bool inclusive, int &stacktop)
{
	if(zone == 0)
		return;

	if(o < order)
	{
		if(zone >= 3)
		{
			int sdist = 2 * (order - o); // the "bit-shift distance" between map orders
			pixset.append(pix << sdist, (pix+1) << sdist);
		}
		else // zone >= 1
		{
			for(int i = 0; i < 4; ++i)
			{
				Pair tmp(4 * pix + 3 - i, o + 1);
				stk.push_back(tmp); // add children
			}
		}
	}
	else if(o > order) // this implies that inclusive == true
	{
		if(zone >= 2) // pixel center in shape
		{
			pixset.append(pix >> (2 * (o - order)));
			stk.resize(stacktop);
		}
		else // (zone == 1): pixel center in safety range
		{
			if(o < omax) // check sublevels
			{
				for(int i = 0; i < 4; ++i)
				{
					Pair tmp(4 * pix + 3 - i, o + 1);
					stk.push_back(tmp);
				}
			}
			else // at resolution limit
			{
				pixset.append(pix >> (2 * (o - order)));
				stk.resize(stacktop);
			}
		}
	}
	else // o == order
	{
		if(zone >= 2)
			pixset.append(pix);
		else if(inclusive) // and (zone >= 1)
		{
			if(order < omax) // check sublevels
			{
				stacktop = stk.getSize(); // remember current stack position
				for(int i = 0; i < 4; ++i)
				{
					Pair tmp(4 * pix + 3 - i, o + 1);
					stk.push_back(tmp);
				}
			}
			else // at resolution limit
				pixset.append(pix);
		}
	}
}

	__device__
int begin_index(int key, int *pix, int N)
{
	int low = 0;
	int high = N - 1;
	while(low <= high)
	{
		int mid = (low + high) >> 1;
		int midVal = pix[mid];
		if(midVal < key)
			low = mid + 1;
		else if(midVal > key)
		{
			if(mid != 0 && pix[mid-1] < key)
				return mid;
			if(mid == 0)
				return mid;
			high = mid - 1;
		}
		else
		{
			if(mid != 0 && pix[mid-1] == key)
				high = mid - 1;
			else
				return mid;
		}
	}
	return -1;
}

