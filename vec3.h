#ifndef VEC3_H
#define VEC3_H

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include "constants.h"
using namespace std;

class Vec3
{
  public:
	double x;
	double y;
	double z;
	__device__ Vec3();
	__device__ Vec3(double x_, double y_, double z_);
	__device__ Vec3(double z_, double phi_);
	__device__ void Normalize();
	__device__ double length();
};

#endif
