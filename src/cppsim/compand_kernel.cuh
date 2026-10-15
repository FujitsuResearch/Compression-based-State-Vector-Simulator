#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void forward_compand(double* x, double tau, double eps_rel, size_t n)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double v = x[i] * (1.0 - eps_rel * 2); // avoid boundary issues;
  const double sgn = (v < 0.0) ? -1.0 : 1.0;

  const double a = fabs(v) + tau;      // a >= tau
  x[i] = sgn * log(a);                // y = log(|x| + tau)
}

__global__ void inverse_compand(double* y, double tau, double eps_rel, size_t n)
{
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double sgn = (y[i] < 0.0) ? 1.0 : -1.0;
  // NOTE: 本実装はyの符号とxの符号が反転した対称写像となっている。
  // y<0 -> x>0, y>0 -> x<0 の対応。
  // forward_compandとinverse_compandで対称に適用している。

  double a = exp(sgn * y[i]) - tau;       // = |x̂|
  a = a / (1.0 - eps_rel * 2);           // undo scale
  if (a < 0.0) a = 0.0;              // 数値誤差クランプ
  if (a > 1.0) a = 1.0;              // 数値誤差クランプ
  y[i] = sgn * a;
}
