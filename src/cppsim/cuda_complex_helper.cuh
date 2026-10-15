#ifndef CUDA_COMPLEX_HELPER_CUH
#define CUDA_COMPLEX_HELPER_CUH

#include <cuComplex.h>
#include <complex>

// CUDA複素数型の便利なヘルパー関数

// std::complex<double> と cuDoubleComplex の相互変換
// Host版 - std::complexのconstexpr関数を使用可能
__host__ inline cuDoubleComplex to_cuda_complex(const std::complex<double>& z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}

__host__ inline std::complex<double> from_cuda_complex(const cuDoubleComplex& z) {
    return std::complex<double>(cuCreal(z), cuCimag(z));
}

// Device版 - 直接値を受け取る
__device__ inline cuDoubleComplex make_cuda_complex_device(double real, double imag) {
    return make_cuDoubleComplex(real, imag);
}

// double配列からcuDoubleComplex配列への変換（in-place）
__host__ __device__ inline cuDoubleComplex* double_array_to_complex_array(double* arr) {
    return reinterpret_cast<cuDoubleComplex*>(arr);
}

__host__ __device__ inline const cuDoubleComplex* double_array_to_complex_array(const double* arr) {
    return reinterpret_cast<const cuDoubleComplex*>(arr);
}

// CPPCTYPE (std::complex<double>) 配列からcuDoubleComplex配列への変換
__host__ __device__ inline cuDoubleComplex* cppctype_to_cuda_complex(std::complex<double>* arr) {
    return reinterpret_cast<cuDoubleComplex*>(arr);
}

__host__ __device__ inline const cuDoubleComplex* cppctype_to_cuda_complex(const std::complex<double>* arr) {
    return reinterpret_cast<const cuDoubleComplex*>(arr);
}

// nvCOMP互換: cuDoubleComplexをdouble*としてnvCOMPに渡す
__host__ __device__ inline double* complex_to_nvcomp_input(cuDoubleComplex* complex_arr) {
    return reinterpret_cast<double*>(complex_arr);
}

__host__ __device__ inline const double* complex_to_nvcomp_input(const cuDoubleComplex* complex_arr) {
    return reinterpret_cast<const double*>(complex_arr);
}

// nvCOMP出力からcuDoubleComplexへの変換
__host__ __device__ inline cuDoubleComplex* nvcomp_output_to_complex(double* double_arr) {
    return reinterpret_cast<cuDoubleComplex*>(double_arr);
}

// 複素数演算用のヘルパー関数
__host__ __device__ inline cuDoubleComplex complex_mul_add(
    cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
    // a * b + c を計算
    return cuCadd(cuCmul(a, b), c);
}

__host__ __device__ inline double complex_abs_squared(cuDoubleComplex z) {
    double real = cuCreal(z);
    double imag = cuCimag(z);
    return real * real + imag * imag;
}

__host__ __device__ inline double complex_abs(cuDoubleComplex z) {
    return sqrt(complex_abs_squared(z));
}

// 複素数行列とベクトルの積
__host__ __device__ inline cuDoubleComplex matrix_vector_element(
    const cuDoubleComplex* matrix, const cuDoubleComplex* vector, 
    int row, int matrix_size) {
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int col = 0; col < matrix_size; col++) {
        result = cuCadd(result, cuCmul(matrix[row * matrix_size + col], vector[col]));
    }
    return result;
}

// 複素数のスカラー倍(演算子オーバーロード)
__host__ __device__ inline cuDoubleComplex operator*(
    const cuDoubleComplex& z, double scalar) {
    return make_cuDoubleComplex(cuCreal(z) * scalar, cuCimag(z) * scalar);
}

__host__ __device__ inline cuDoubleComplex operator*(
    double scalar, const cuDoubleComplex& z) {
    return make_cuDoubleComplex(cuCreal(z) * scalar, cuCimag(z) * scalar);
}

// ブロック内の全要素がゼロかどうかをチェックするカーネル
__global__ void check_all_zero_kernel(const cuDoubleComplex* decomp_block, 
        unsigned long namps_per_block, bool* is_all_zero) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < namps_per_block) {
        cuDoubleComplex amp = decomp_block[idx];
        // complex_abs_squared を使用してより効率的にゼロチェック
        if (complex_abs_squared(amp) != 0.0) {
            *is_all_zero = false;
        }
    }
}

#endif // CUDA_COMPLEX_HELPER_CUH