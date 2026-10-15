// compand_bitcomp_pipeline.hpp
#include "bitcomp_adapter.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include "check_util.cuh"
#include "compand_kernel.cuh"
#include <cuComplex.h>

// ====== 圧縮パイプライン（前段）：log圧伸 → Bitcomp lossy 圧縮 ======
inline size_t forward_compand_and_bitcomp_compress(
    cuDoubleComplex* d_decomp_block, // Input: decompressed SV block (device)
    size_t dim_double,                     // Input: dimension in double (＝ 2 × number of complex)
    void* d_comp_block,                    // Output: compressed SV block (device)
    bitcompHandle_t bitcomp_plan,          // Bitcomp plan (pre-created)
    double eps_rel,                        // Relative error bound
    double eps_abs,                        // Absolute error bound (near zero, set to 0 if not needed)
    cudaStream_t stream
) {
  if (eps_rel <= 0.0) throw std::invalid_argument("eps_rel must be > 0");
  const double gamma = 1.0 - 2.0 * eps_rel;
  const double tau_prime =
      (eps_abs > 0.0) ? (gamma * (eps_abs / eps_rel)) : 0.0;
  const double delta = log1p(eps_rel);                                  // log空間の絶対誤差上限
  
  dim3 blk(256), grd((dim_double + blk.x - 1)/blk.x);
  forward_compand<<<grd, blk, 0, stream>>>(reinterpret_cast<double*>(
    d_decomp_block), tau_prime, eps_rel, dim_double);
  
  BITCOMP_CHECK(bitcompSetStream(bitcomp_plan, stream));
  BITCOMP_CHECK(bitcompCompressLossy_fp64(
          bitcomp_plan,
          reinterpret_cast<double*>(d_decomp_block),
          d_comp_block,
          delta));
  
  size_t comp_bytes = 0;
  BITCOMP_CHECK(bitcompGetCompressedSize(d_comp_block, &comp_bytes));
  return comp_bytes;
}

// ====== 伸長パイプライン（後段）：Bitcomp 伸長 → 逆圧伸 ======
inline void bitcomp_decompress_and_inverse_compand(
    void* d_comp_block,        // Input: compressed SV block (device)
    size_t dim_double,               // Input: dimension in double (＝ 2 × number of complex)
    cuDoubleComplex* d_decomp_block, // Output: decompressed SV block (device)
    bitcompHandle_t bitcomp_plan,    // Bitcomp plan (pre-created)
    double tau,
    double eps_rel,
    cudaStream_t stream
) {
  BITCOMP_CHECK(bitcompSetStream(bitcomp_plan, stream));
  BITCOMP_CHECK(bitcompUncompress(bitcomp_plan, d_comp_block, 
    reinterpret_cast<double*>(d_decomp_block)));
  
  dim3 blk(256), grd((dim_double + blk.x - 1)/blk.x);  
  inverse_compand<<<grd, blk, 0, stream>>>(
    reinterpret_cast<double*>(d_decomp_block), tau, eps_rel, dim_double);
}