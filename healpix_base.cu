#include "healpix_base.h"

  __device__
Healpix_Base::Healpix_Base()
{
  order = 13;
  Nside = 1 << 13;
  Npface = Nside << order;
  Ncap = (Npface - Nside) << 1;
  Npix = 12 * Npface;
  fact2 = 4.0 / Npix;
  fact1 = (Nside << 1) * fact2;
}

  __device__
Healpix_Base::Healpix_Base(int order_)
{
  order = order_;
  Nside = 1 << order;
  Npface = Nside << order;
  Ncap = (Npface - Nside) << 1;
  Npix = 12 * Npface;
  fact2 = 4.0 / Npix;
  fact1 = (Nside << 1) * fact2;
}

  __device__
void Healpix_Base::set_order(int order_)
{
  order = order_;
  Nside = 1 << order;
  Npface = Nside << order;
  Ncap = (Npface - Nside) << 1;
  Npix = 12 * Npface;
  fact2 = 4.0 / Npix;
  fact1 = (Nside << 1) * fact2;
}
  __device__ 
int Healpix_Base::zphi2pix(double z,double phi)
{
  double za = fabs(z);
  double tt = fmodulo(phi * inv_halfpi,4.0);

  if(za <= twothird) // Equatorial Region
  {
	double temp1 = Nside * (0.5 + tt);
	double temp2 = Nside * (z * 0.75);

	int jp = int(temp1 - temp2); //index of ascending edge line
	int jm = int(temp1 + temp2); //index of descending edge line

	int ifp = jp >> order;
	int ifm = jm >> order;

	int face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));

	int ix = jm & (Nside - 1);
	int iy = Nside - (jp & (Nside - 1)) - 1;
	return xyf2nest(ix,iy,face_num);
  }
  else // Polar region
  {
	int ntt = min(3,(int)tt);
	double tp = tt - ntt;
	double tmp = Nside * sqrt(3 * (1-za));
	int jp = (int) (tp * tmp);
	int jm = (int) ((1.0 - tp) * tmp);
	jp = min(jp, Nside-1);
	jm = min(jm, Nside-1);
	return (z >= 0) ? xyf2nest(Nside - jm - 1, Nside - jp - 1, ntt) : xyf2nest(jp,jm,ntt + 8);
  }
}

  __device__ 
double Healpix_Base::fmodulo(double v1, double v2)
{
  if (v1 >= 0)
	return (v1 < v2) ? v1 : fmod(v1,v2);
  double tmp  = fmod(v1,v2) + v2;
  return (tmp == v2) ? 0.0 : tmp;
}

  __device__ 
int Healpix_Base::spread_bits(int v)
{
  int v1 = v & 0xff;
  int v2 = (v>>8)&0xff;
  return utab(v1) | (utab(v2) << 16);
}

  __device__ 
int Healpix_Base::compress_bits(int v)
{
  int raw = (v&0x5555) | ((v&0x55550000)>>15);
  return ctab(raw&0xff) | (ctab(raw>>8) << 4);
}

  __device__ 
int Healpix_Base::xyf2nest(int ix,int iy,int face_num)
{
  return (face_num << (2 * order)) + spread_bits(ix) + (spread_bits(iy) << 1);
}

  __device__
void Healpix_Base::nest2xyf(int pix,int &ix,int &iy,int &face_num)
{
  face_num = pix >> (2 * order);
  pix &= (Npface-1);
  ix = compress_bits(pix);
  iy = compress_bits(pix >> 1);
}

  __device__ 
void Healpix_Base::pix2loc(int pix, double &z, double &phi, double &sth, bool &have_sth)
{
  int jrll[12] = {2,2,2,2,3,3,3,3,4,4,4,4};
  int jpll[12] = {1,3,5,7,0,2,4,6,1,3,5,7};
  
  have_sth = false;
  int face_num, ix, iy;
  nest2xyf(pix,ix,iy,face_num);

  int jr = ((jrll[face_num]) << order) - ix - iy - 1;

  int nr;

  if(jr < Nside)
  {
	nr = jr;
	double tmp = (nr * nr) * fact2;
	z = 1 - tmp;
	if (z > 0.99)
	{
	  sth = sqrt(tmp * (2.0 - tmp));
	  have_sth = true;
	}
  }
  else if(jr > 3 * Nside)
  {
	nr = Nside * 4 - jr;
	double tmp = (nr * nr) * fact2;
	z = tmp - 1;
	if(z < -0.99)
	{
	  sth = sqrt(tmp * (2.0 - tmp));
	  have_sth = true;
	}
  }
  else
  {
	nr = Nside;
	z = (2 * Nside - jr) * fact1;
  }


  int tmp = jpll[face_num] * nr + ix - iy;
  if(tmp < 0)
	tmp += 8 * nr;
  phi = (nr == Nside) ? 0.75 * halfpi * tmp * fact1 : (0.5 * halfpi * tmp) / nr;
  return;
}

  __device__ 
void Healpix_Base::pix2zphi(int pix, double &z, double &phi)
{
  bool dum_b;
  double dum_d;
  pix2loc(pix,z,phi,dum_d,dum_b);
}

  __device__ 
double Healpix_Base::max_pixrad()
{
  Vec3 va(2.0 / 3.0, pi / (4 * Nside));
  double t1 = 1.0 - 1.0 / Nside;
  t1 *= t1;
  Vec3 vb(1 - t1 / 3, 0);
  return v_angle(va,vb);
}



