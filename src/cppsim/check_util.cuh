// ====== ユーティリティ ======
#define CUDA_CHECK(expr) do { \
  cudaError_t _err = (expr); \
  if (_err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ")+cudaGetErrorString(_err)+" @ " __FILE__ ":" + std::to_string(__LINE__)); \
  } \
} while(0)

#define BITCOMP_CHECK(expr) do { \
  int _st = (expr); \
  if (_st != 0) { \
    throw std::runtime_error(std::string("bitcomp error code ")+std::to_string(_st)+" @ " __FILE__ ":" + std::to_string(__LINE__)); \
  } \
} while(0)
