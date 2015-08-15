#include "vec3.h"

__device__ Vec3::Vec3()
{
  x = 0;
  y = 0;
  z = 0;
}
__device__ Vec3::Vec3(double x_, double y_, double z_)
{
  x = x_;
  y = y_;
  z = z_;
}
__device__ Vec3::Vec3(double z_, double phi_)
{
  double sinTheta = sqrt((1 - z_) * (1 + z_));
  x = sinTheta * cos(phi_);
  y = sinTheta * sin(phi_);
  z = z_;
}
__device__ void Vec3::Normalize()
{
  double l = 1 / sqrt(x * x + y * y + z * z);
  x *= l;
  y *= l;
  z *= l;
}
__device__ double Vec3::length()
{
  return sqrt(x * x + y * y + z * z);
}
