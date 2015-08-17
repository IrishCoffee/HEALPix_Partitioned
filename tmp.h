#ifndef HEALPIX_BASE_H
#define HEALPIX_BASE_H

#include "vec3.h"
#include "geometry_cal.h"
#include <thrust/device_vector.h>

#define utab(m) (int)( (m&0x1) | ((m&0x2) << 1) |  ((m&0x4) << 2) |  ((m&0x8) << 3) | ((m&0x10) << 4) | ((m&0x20) << 5) | ((m&0x40) << 6) | ((m&0x80) << 7))
#define ctab(m) (int)( (m&0x1) | ((m&0x2) << 7) |  ((m&0x4) >> 1) |  ((m&0x8) << 6) | ((m&0x10) >> 2) | ((m&0x20) << 5) | ((m&0x40) >> 3) | ((m&0x80) << 4))

struct Point
{
  double z;
  double phi;
  double x,y;

  __device__ Point(double z,double phi):
	z(z),
	phi(phi),
	x(sqrt((1-z)*(1+z)) * cos(phi)),
	y(sqrt((1-z)*(1+z)) * sin(phi))
  {}
};
//__device__ int jrll[12] = {2,2,2,2,3,3,3,3,4,4,4,4};
//__device__ int jpll[12] = {1,3,5,7,0,2,4,6,1,3,5,7};

class Healpix_Base
{
  public:
	__device__ Healpix_Base();
	__device__ Healpix_Base(int order_);
	__device__ void set_order(int order_);
	__device__ int zphi2pix(double z, double phi);
	__device__ double fmodulo(double v1, double v2);
	__device__ int spread_bits(int v);
	__device__ int compress_bits(int v);
	__device__ int xyf2nest(int ix,int iy, int face_num);
	__device__ void nest2xyf(int pix, int &ix, int &iy, int &face_num);
	__device__ void pix2loc(int pix, double &z, double &phi, double &sth, bool &have_sth);
	__device__ void pix2zphi(int pix, double &z, double &phi);
	__device__ double max_pixrad();
  
	int order;
	int Nside;
	int Npix;
	int Npface;
	int Ncap;
	double fact2;
	double fact1;
};


#endif
