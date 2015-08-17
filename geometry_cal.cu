#include "geometry_cal.h"

__host__ __device__ double radians(double degree)
{
  return degree * pi / 180.0;
}

__device__ double dotProduct(Vec3 &v1, Vec3 &v2)
{
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ void crossProduct(Vec3 &v1, Vec3 &v2, Vec3 &v)
{
  v.x = v1.y * v2.z - v1.z * v2.y;
  v.y = v1.z * v2.x - v1.x * v2.z;
  v.z = v1.x * v2.y - v1.y * v2.x;
  return;
}

__device__ double v_angle(Vec3 &v1, Vec3 &v2)
{
 	Vec3 v;
	crossProduct(v1,v2,v);
	double a = v.length();
	double b = dotProduct(v1,v2);
	return atan2(a,b);
}

__device__ double cosdist_zphi(double z1, double phi1, double z2, double phi2)
{
  return z1 * z2 + cos(phi1 - phi2) * sqrt((1.0 - z1 * z1) * (1.0 - z2 * z2));
}

__device__
bool matched(double ra1,double dec1,double ra2,double dec2,double radius)
{
  double z1 = sin(radians(dec1));
  double x1 = cos(radians(dec1)) * cos(radians(ra1));
  double y1 = cos(radians(dec1)) * sin(radians(ra1));

  double z2 = sin(radians(dec2));
  double x2 = cos(radians(dec2)) * cos(radians(ra2));
  double y2 = cos(radians(dec2)) * sin(radians(ra2));

  double distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
  double dist2 = 4 * pow(sin(radians(0.0056 / 2)),2);
 
  if(distance <= dist2)
	return true;
  return false;

}


