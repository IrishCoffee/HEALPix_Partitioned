#ifndef GEOMETRY_CAL_H
#define GEOMETRY_CAL_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>
#include "vec3.h"

__host__ __device__ double radians(double degree);
__device__ double dotProduct(Vec3 &v1, Vec3 &v2);
__device__ void crossProduct(Vec3 &v1, Vec3 &v2,Vec3 &v);
__device__ double v_angle(Vec3 &v1, Vec3 &v2);
__device__ double cosdist_zphi(double z1, double phi1,double z2, double phi2);

__device__ bool matched(double ra1,double dec1,double ra2,double dec2,double radius);

#endif
