#include "compressed_state.hpp"
#include "compand_bitcomp_pipeline.cuh"
// #include "gpu_pointwise_compressor.cuh"
#include <cuComplex.h>
#include "cuda_complex_helper.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#ifdef _USE_RMM
// RMM (Rapids Memory Manager) headers
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#endif

// Define static member variables declared in compressed_state.hpp
cuDoubleComplex* CompressedQuantumState::d_1q_matrix = nullptr;
cuDoubleComplex* CompressedQuantumState::d_2q_matrix = nullptr;
#ifdef _USE_RMM
std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> CompressedQuantumState::pool_mr = nullptr;
#endif

size_t get_padded_bytes(size_t original_bytes) {
    size_t padding_bytes = (8 - (original_bytes % 8)) % 8; // padding for 8-byte alignment
    return original_bytes + padding_bytes;
}

template <typename T>
T* device_alloc_bytes(size_t bytes) {
#ifdef _USE_RMM
    auto mr = rmm::mr::get_current_device_resource();
    return static_cast<T*>(mr->allocate(bytes));
#else
    T* ptr = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    return ptr;
#endif
}

// Keep allocation logic unified across CUDA malloc/free and RMM backends.
inline void device_free_bytes(void* ptr, size_t bytes) {
#ifdef _USE_RMM
    auto mr = rmm::mr::get_current_device_resource();
    mr->deallocate(ptr, bytes);
#else
    cudaFree(ptr);
#endif
}

char* CompressedQuantumState::compress_SV_block_GPU(
    CPPCTYPE* decomp_block,
    size_t &comp_bytes
) {
    size_t decomp_bytes = namps_per_block * sizeof(cuDoubleComplex);
    cuDoubleComplex *d_decomp_block;
    char* d_comp_block;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate GPU memory and copy decompressed block to device
    d_decomp_block = device_alloc_bytes<cuDoubleComplex>(decomp_bytes);
    d_comp_block = device_alloc_bytes<char>(decomp_bytes); // Todo: size tuning
    cudaMemcpy(d_decomp_block, decomp_block, decomp_bytes, cudaMemcpyHostToDevice);
    
    // Compress the block
    comp_bytes = forward_compand_and_bitcomp_compress(
        d_decomp_block,      // Input: decompressed SV block (device)
        namps_per_block * 2, // Input: dimension in double
        d_comp_block,        // Output: compressed SV block (device)
        bitcomp_plan,        // Bitcomp plan (pre-created)
        bitcomp_eps_rel, 
        bitcomp_eps_abs, 
        stream);
    cudaStreamSynchronize(stream);

    // Copy compressed block back to host
    comp_bytes = get_padded_bytes(comp_bytes);
    char* comp_block;
    // cudaMallocHost((void**)&comp_block, comp_bytes);
    comp_block = (char*)malloc(comp_bytes);
    cudaMemcpy(comp_block, d_comp_block, comp_bytes, cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    device_free_bytes(d_decomp_block, decomp_bytes);
    device_free_bytes(d_comp_block, decomp_bytes);
    cudaStreamDestroy(stream);

    return comp_block;
}

CPPCTYPE* CompressedQuantumState::decompress_SV_block_GPU(
    char* comp_block,
    size_t comp_bytes
) {
    // Allocate memory for decompressed block
    size_t decomp_bytes = namps_per_block * sizeof(cuDoubleComplex);
    CPPCTYPE* decomp_block = (CPPCTYPE*)malloc(decomp_bytes);

    if (comp_block == nullptr) {
        // Make all-zero block
        memset(decomp_block, 0, decomp_bytes);
        return decomp_block;
    }

    char* d_comp_block;
    cuDoubleComplex* d_decomp_block;
    const double gamma = (1.0 - 2.0 * bitcomp_eps_rel);
    const double tau_prime = gamma * (bitcomp_eps_abs / bitcomp_eps_rel);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate GPU memory and copy compressed block to device
    d_comp_block = device_alloc_bytes<char>(comp_bytes);
    d_decomp_block = device_alloc_bytes<cuDoubleComplex>(decomp_bytes);
    cudaMemcpy(d_comp_block, comp_block, comp_bytes, cudaMemcpyHostToDevice);

    // Decompress the block
    bitcomp_decompress_and_inverse_compand(
        d_comp_block,        // Input: compressed SV block (device)
        namps_per_block * 2, // Input: dimension in double
        d_decomp_block,      // Output: decompressed SV block (device)
        bitcomp_plan,        // Bitcomp plan (pre-created)
        tau_prime,
        bitcomp_eps_rel,
        stream
    );
    
    // Copy decompressed block back to host
    cudaMemcpy(decomp_block, d_decomp_block, decomp_bytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    device_free_bytes(d_comp_block, comp_bytes);
    device_free_bytes(d_decomp_block, decomp_bytes);
    cudaStreamDestroy(stream);

    return decomp_block;
}

// GPU-based processing of a group of SV blocks
void CompressedQuantumState::process_SV_group_GPU(
    const std::vector<UINT>& gate_ids, 
    const std::vector<UINT>& block_group,
    int PU_id,
    double& decomp_time, double& gate_time, double& comp_time,
    double& hmalloc_time, double& dmalloc_time, double& h2d_time, double& d2h_time, 
    double& free_time, double& size_time
) {
    ITYPE current_offset;
    std::vector<UINT> my_block_ids;
#ifdef _USE_MPI
    MPIutil& mpiutil = MPIutil::get_inst();
    UINT rank = mpiutil.get_rank();
    for (auto block_id : block_group) {
        if (block_id / nblocks == rank) {
            my_block_ids.push_back(block_id % nblocks);
        }
    }
#else
    my_block_ids = block_group;
#endif
    UINT my_nblocks = my_block_ids.size();  
    std::vector<size_t> comp_block_bytes_vector(my_nblocks);

    // Set GPU device based on PU_id
    cudaSetDevice(PU_id);

    // Calculate total size for compressed SV blocks in the group
    auto ts_size = std::chrono::steady_clock::now();
    ITYPE total_comp_bytes = 0ULL;
    for (UINT i = 0; i < my_nblocks; i++) {
        UINT block_id = my_block_ids[i];
        if (block_pointers[block_id] == nullptr) {
            comp_block_bytes_vector[i] = 0UL;
        } else {
            comp_block_bytes_vector[i] = blocks_bytes[block_id];
            total_comp_bytes += static_cast<ITYPE>(blocks_bytes[block_id]);
        }
    }
    auto te_size = std::chrono::steady_clock::now();
    auto t_size = std::chrono::duration_cast<std::chrono::nanoseconds>(te_size - ts_size);
    size_time += t_size.count() / 1e9;

    auto ts_dmalloc = std::chrono::steady_clock::now();
    // Allocate GPU memory for compressed SV blocks (input) in the group
    char *d_comp_blocks_in;
    d_comp_blocks_in = device_alloc_bytes<char>(total_comp_bytes);

    // Allocate and initialize GPU memory for decompressed SV blocks (tmp) in the group
    cuDoubleComplex *d_decomp_blocks;
    ITYPE total_decomp_bytes = my_nblocks * namps_per_block * sizeof(cuDoubleComplex);
    d_decomp_blocks = device_alloc_bytes<cuDoubleComplex>(total_decomp_bytes);
    cudaMemset(*(void**)&d_decomp_blocks, 0, total_decomp_bytes);

    // Allocate GPU memory for compressed SV blocks (output) in the group
    char *d_comp_blocks_out;
    d_comp_blocks_out = device_alloc_bytes<char>(total_decomp_bytes); // Todo: size tuning
    auto te_dmalloc = std::chrono::steady_clock::now();
    auto t_dmalloc = std::chrono::duration_cast<std::chrono::nanoseconds>(te_dmalloc - ts_dmalloc);
    dmalloc_time += t_dmalloc.count() / 1e9;

    // Call GPU kernel for the SV Group
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy compressed SV blocks in the group from CPU to GPU memory
    auto ts_h2d = std::chrono::steady_clock::now();
    current_offset = 0;
    for (UINT i = 0; i < my_nblocks; i++) {
        UINT block_id = my_block_ids[i];
        if (comp_block_bytes_vector[i] == 0) {
            continue;
        }
        cudaMemcpyAsync(d_comp_blocks_in + current_offset, block_pointers[block_id], 
            comp_block_bytes_vector[i], cudaMemcpyHostToDevice, stream);
        current_offset += static_cast<ITYPE>(comp_block_bytes_vector[i]);
    }
    cudaStreamSynchronize(stream);
    auto te_h2d = std::chrono::steady_clock::now();
    auto t_h2d = std::chrono::duration_cast<std::chrono::nanoseconds>(te_h2d - ts_h2d);
    h2d_time += t_h2d.count() / 1e9;

    // Prepare parameters used in decompression and compression    
    const double gamma = (1.0 - 2.0 * bitcomp_eps_rel);
    const double tau_prime = gamma * (bitcomp_eps_abs / bitcomp_eps_rel);

    auto ts_decomp = std::chrono::steady_clock::now();
    current_offset = 0;
    for(UINT i = 0; i < my_nblocks; i++){
        if (comp_block_bytes_vector[i] == 0) {
            continue;
        }
        bitcomp_decompress_and_inverse_compand(
            d_comp_blocks_in + current_offset,        // Input : compressed SV block (device)
            namps_per_block * 2,                      // Input : dimension in double
            d_decomp_blocks + (i * namps_per_block),  // Output: decompressed SV block (device)
            bitcomp_plan,                             // Bitcomp plan (pre-created)
            tau_prime,
            bitcomp_eps_rel,                                                
            stream
        );
        current_offset += static_cast<ITYPE>(comp_block_bytes_vector[i]);
    }
    cudaStreamSynchronize(stream);
    auto te_decomp = std::chrono::steady_clock::now();
    auto t_decomp = std::chrono::duration_cast<std::chrono::nanoseconds>(te_decomp - ts_decomp);
    decomp_time += t_decomp.count() / 1e9;

    // Apply gate operations on all SV blocks in the group
    auto ts_gate = std::chrono::steady_clock::now();
    for (auto gate_id : gate_ids) {
        auto gate = gate_list[gate_id];
        ComplexMatrix matrix = gate->get_matrix();
        // apply the gate to the decompressed blocks
        if (gate->get_name() == "CNOT") {
            apply_cnot_GPU(gate, my_block_ids, static_cast<void*>(d_decomp_blocks), stream);
        } else if (matrix.size() == 4) {
            apply_1q_gate_GPU(gate, matrix, d_1q_matrix, my_block_ids, 
                static_cast<void*>(d_decomp_blocks), stream);
        } else if (matrix.size() == 16) {
            apply_2q_gate_GPU(gate, matrix, d_2q_matrix, my_block_ids, 
                static_cast<void*>(d_decomp_blocks), stream);
        } else {
            throw NotImplementedException(
                "Error: CompressedQuantumState::process_SV_group: "
                "only single- or two-qubit gates are implemented");
        }
        cudaStreamSynchronize(stream);
    }
    auto te_gate = std::chrono::steady_clock::now();
    auto t_gate = std::chrono::duration_cast<std::chrono::nanoseconds>(te_gate - ts_gate);
    gate_time += t_gate.count() / 1e9;

    // Compress all SV Blocks in the Group
    auto ts_comp = std::chrono::steady_clock::now();
    current_offset = 0;    
    for(UINT i = 0; i < my_nblocks; i++){
        cuDoubleComplex *d_decomp_block = d_decomp_blocks + (i * namps_per_block);

        if (comp_block_bytes_vector[i] == 0) {            
            // Check if the decompressed block is still all-zero on GPU
            auto ts_zero_check = std::chrono::steady_clock::now();
            bool is_all_zero;
            bool *d_is_all_zero;
            d_is_all_zero = device_alloc_bytes<bool>(sizeof(bool));
            cudaMemset(d_is_all_zero, true, sizeof(bool));
            
            dim3 threadsPerBlock(256);
            dim3 blocksPerGrid((namps_per_block + threadsPerBlock.x - 1) / threadsPerBlock.x);     
            check_all_zero_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_decomp_block, namps_per_block, d_is_all_zero);

            cudaMemcpy(&is_all_zero, d_is_all_zero, sizeof(bool), cudaMemcpyDeviceToHost);
            device_free_bytes(d_is_all_zero, sizeof(bool));
            auto te_zero_check = std::chrono::steady_clock::now();
            auto t_zero_check = std::chrono::duration_cast<std::chrono::nanoseconds>(te_zero_check - ts_zero_check);
            total_zero_check_time += t_zero_check.count() / 1e9;

            if (is_all_zero) {
                comp_block_bytes_vector[i] = 0UL;
                continue;
            }
        }

        size_t comp_bytes = forward_compand_and_bitcomp_compress(
            d_decomp_block,                     // Input : decompressed SV Blocks
            namps_per_block * 2,                // Input : dimension in double 
            d_comp_blocks_out + current_offset, // Output: compressed SV Block
            bitcomp_plan,                       // Bitcomp plan (pre-created)
            bitcomp_eps_rel,
            bitcomp_eps_abs,
            stream
        );
        
        comp_block_bytes_vector[i] = get_padded_bytes(comp_bytes);                
        current_offset += static_cast<ITYPE>(comp_block_bytes_vector[i]);
    }
    cudaStreamSynchronize(stream);
    auto te_comp = std::chrono::steady_clock::now();
    auto t_comp = std::chrono::duration_cast<std::chrono::nanoseconds>(te_comp - ts_comp);
    comp_time += t_comp.count() / 1e9;

    auto ts_hmalloc = std::chrono::steady_clock::now();
    current_offset = 0;    
    for (UINT i = 0; i < my_nblocks; i++) {
        UINT block_id = my_block_ids[i];
    
        if (block_pointers[block_id] != nullptr) {
            // Free existing memory
            // cudaFreeHost(block_pointers[block_id]);
            free(block_pointers[block_id]);
        }
        
        if (comp_block_bytes_vector[i] == 0) {
            block_pointers[block_id] = nullptr;
            blocks_bytes[block_id] = 0UL;
            continue;
        }
    
        // cudaMallocHost((void**)&block_pointers[block_id], comp_block_bytes_vector[i]);
        block_pointers[block_id] = (char*)malloc(comp_block_bytes_vector[i]);        
        blocks_bytes[block_id] = comp_block_bytes_vector[i];
        current_offset += static_cast<ITYPE>(comp_block_bytes_vector[i]);
    }
    cudaStreamSynchronize(stream);    
    auto te_hmalloc = std::chrono::steady_clock::now();
    auto t_hmalloc = std::chrono::duration_cast<std::chrono::nanoseconds>(te_hmalloc - ts_hmalloc);
    hmalloc_time += t_hmalloc.count() / 1e9;
    
    auto ts_d2h = std::chrono::steady_clock::now();
    current_offset = 0;    
    for (UINT i = 0; i < my_nblocks; i++) {
        UINT block_id = my_block_ids[i];
        
        if (comp_block_bytes_vector[i] == 0) {
            continue;
        }
    
        // Copy back the compressed SV Block from GPU to CPU memory
        // cudaMallocHost((void**)&block_pointers[block_id], comp_block_bytes_vector[i]);
        cudaMemcpyAsync(block_pointers[block_id], d_comp_blocks_out + current_offset, 
            comp_block_bytes_vector[i], cudaMemcpyDeviceToHost, stream);    
        current_offset += static_cast<ITYPE>(comp_block_bytes_vector[i]);
    }
    cudaStreamSynchronize(stream);    
    auto te_d2h = std::chrono::steady_clock::now();
    auto t_d2h = std::chrono::duration_cast<std::chrono::nanoseconds>(te_d2h - ts_d2h);
    d2h_time += t_d2h.count() / 1e9;
    
    auto ts_free = std::chrono::steady_clock::now();
    cudaStreamDestroy(stream);    
    // Free GPU memory
    device_free_bytes(d_comp_blocks_in, total_comp_bytes);
    device_free_bytes(d_decomp_blocks, total_decomp_bytes);
    device_free_bytes(d_comp_blocks_out, total_decomp_bytes);
    auto te_free = std::chrono::steady_clock::now();
    auto t_free = std::chrono::duration_cast<std::chrono::nanoseconds>(te_free - ts_free);
    free_time += t_free.count() / 1e9;
}

// // GPU側で使う1qubit gate行列
// __constant__ double gate_1q[][4] = {
//     {1,0,0,0}, // I
//     {0,1,1,0}, // X
//     {1,0,0, -1}, // Z
//     {1/sqrt(2),0,0,1/sqrt(2), 0,0,0,0, 0,0,1/sqrt(2),0, 1/sqrt(2),0,0, -1/sqrt(2)}, // H
//     {1/sqrt(2),0,0,-1/sqrt(2), 0,0,0,0, 0,0,1/sqrt(2),0, 1/sqrt(2),0,0,1/sqrt(2)} // Y
// };

__global__ void process_1qubit_kernel_sameblock(
    cuDoubleComplex* d_decomp_block, cuDoubleComplex* d_matrix, unsigned int namps_per_block,
    ITYPE mask_low, ITYPE mask_high, ITYPE mask
) {
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_index < namps_per_block / 2) {        
        ITYPE amp_offset0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
        ITYPE amp_offset1 = amp_offset0 | mask;
        cuDoubleComplex a0 = d_decomp_block[amp_offset0];
        cuDoubleComplex a1 = d_decomp_block[amp_offset1];
        cuDoubleComplex matrix00 = d_matrix[0];
        cuDoubleComplex matrix01 = d_matrix[1];
        cuDoubleComplex matrix10 = d_matrix[2];
        cuDoubleComplex matrix11 = d_matrix[3];
        d_decomp_block[amp_offset0] = cuCadd(cuCmul(matrix00,a0), cuCmul(matrix01,a1));
        d_decomp_block[amp_offset1] = cuCadd(cuCmul(matrix10,a0), cuCmul(matrix11,a1));
    }
}

__global__ void process_1qubit_kernel_diffblock(
    cuDoubleComplex* d_decomp_block0, cuDoubleComplex* d_decomp_block1,
    cuDoubleComplex* d_matrix, UINT namps_per_block              
) {
    ITYPE amp_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (amp_offset < namps_per_block) {
        cuDoubleComplex a0 = d_decomp_block0[amp_offset];
        cuDoubleComplex a1 = d_decomp_block1[amp_offset];
        cuDoubleComplex matrix00 = d_matrix[0];
        cuDoubleComplex matrix01 = d_matrix[1];
        cuDoubleComplex matrix10 = d_matrix[2];
        cuDoubleComplex matrix11 = d_matrix[3];
        cuDoubleComplex new_a0 = cuCadd(cuCmul(matrix00, a0), cuCmul(matrix01, a1));
        cuDoubleComplex new_a1 = cuCadd(cuCmul(matrix10, a0), cuCmul(matrix11, a1));
        d_decomp_block0[amp_offset] = new_a0;
        d_decomp_block1[amp_offset] = new_a1;
    }
}

__global__ void apply_diagonal_element_kernel(cuDoubleComplex* d_decomp_block,
    cuDoubleComplex diagonal_element, UINT namps_per_block
) {
    ITYPE amp_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (amp_offset < namps_per_block) {
        d_decomp_block[amp_offset] = cuCmul(d_decomp_block[amp_offset], diagonal_element);
    }
}

void CompressedQuantumState::apply_1q_gate_GPU(
    QuantumGateMatrix* gate,
    const ComplexMatrix& matrix,
    cuDoubleComplex *d_matrix,
    const std::vector<UINT>& block_ids, 
    void* d_decompressed_SVblocks, 
    cudaStream_t stream
) {
    UINT target_qubit_index = gate->get_target_index_list()[0];
    const ITYPE mask = 1ULL << target_qubit_index;
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    const ITYPE bmask = mask >> block_size;
    ITYPE state_index;

    auto ts_1q = std::chrono::steady_clock::now();

    UINT threadsPerBlock = 256;
    cuDoubleComplex* d_decomp_blocks = static_cast<cuDoubleComplex*>(d_decompressed_SVblocks);

    // ゲート情報をGPUのglobalメモリに転送
    // ここはゲートの種類ごとにindexを振り、適切なconstメモリを呼ぶように実装変更する必要あり
    auto ts_1q_matrix = std::chrono::steady_clock::now();
    std::vector<std::complex<double>> h_matrix(4);
    h_matrix[0] = matrix(0,0);
    h_matrix[1] = matrix(0,1);
    h_matrix[2] = matrix(1,0);
    h_matrix[3] = matrix(1,1);
    // cuDoubleComplex* d_matrix;
    // cudaMalloc((void**)&d_matrix, 4 * sizeof(cuDoubleComplex));
    cudaMemcpy(d_matrix, h_matrix.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    auto te_1q_matrix = std::chrono::steady_clock::now();
    auto t_1q_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(te_1q_matrix - ts_1q_matrix);
    total_1q_matrix_time += t_1q_matrix.count() / 1e9;

    if (target_qubit_index < block_size) {
        // apply the gate within each block
        UINT nelems = namps_per_block / 2;
        UINT blocksPerGrid = (nelems + threadsPerBlock - 1) / threadsPerBlock;

        auto ts_1q_in = std::chrono::steady_clock::now();
        for (UINT i = 0; i < block_ids.size(); i++) {
            cuDoubleComplex* d_decomp_block = d_decomp_blocks + (size_t)i * namps_per_block;
            process_1qubit_kernel_sameblock<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                d_decomp_block, d_matrix, namps_per_block, mask_low, mask_high, mask);            
        }
        cudaStreamSynchronize(stream);
        auto te_1q_in = std::chrono::steady_clock::now();
        auto t_1q_in = std::chrono::duration_cast<std::chrono::nanoseconds>(te_1q_in - ts_1q_in);
        total_1q_in_time += t_1q_in.count() / 1e9;
    } else {
        bool diagonal = false;
        if (matrix(0,1) == 0.0 && matrix(1,0) == 0.0) {
            diagonal = true;
        }
        if (diagonal and (target_qubit_index >= inner_qc)) {
#ifdef _USE_MPI                
            const UINT rank = MPIutil::get_inst().get_rank();
#else
            const UINT rank = 0;
#endif
            const UINT pair_rank_bit = 1UL << (target_qubit_index - inner_qc);
            const UINT isone = (rank & pair_rank_bit) != 0;

            for (UINT i = 0; i < block_ids.size(); i++) {
                cuDoubleComplex* d_decomp_block = d_decomp_blocks + (size_t)i * namps_per_block;
                cuDoubleComplex diagonal_val = 
                    isone ? make_cuDoubleComplex(matrix(1,1).real(), matrix(1,1).imag())
                        : make_cuDoubleComplex(matrix(0,0).real(), matrix(0,0).imag());
                UINT blocksPerGrid = (namps_per_block + threadsPerBlock - 1) / threadsPerBlock;
                apply_diagonal_element_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    d_decomp_block, diagonal_val, namps_per_block);
            }
        } else {
            // apply the gate between pairs of blocks
            UINT blocksPerGrid = (namps_per_block + threadsPerBlock - 1) / threadsPerBlock;            
            std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());

            auto ts_1q_out = std::chrono::steady_clock::now();
            while (target_block_ids.size() > 0) {
                auto ts_1q_out_search = std::chrono::steady_clock::now();
                UINT block_id0 = target_block_ids.front();
                target_block_ids.pop_front();
                UINT block_id1 = block_id0 + bmask;
                target_block_ids.remove(block_id1);
                UINT local_block_id0 = search_index(block_ids, block_id0);
                UINT local_block_id1 = search_index(block_ids, block_id1);
                cuDoubleComplex* d_decomp_block0 = 
                    d_decomp_blocks + (size_t)local_block_id0 * namps_per_block;
                cuDoubleComplex* d_decomp_block1 = 
                    d_decomp_blocks + (size_t)local_block_id1 * namps_per_block;
                auto te_1q_out_search = std::chrono::steady_clock::now();
                auto t_1q_out_search = std::chrono::duration_cast<std::chrono::nanoseconds>(te_1q_out_search - ts_1q_out_search);
                total_1q_out_search_time += t_1q_out_search.count() / 1e9;

                process_1qubit_kernel_diffblock<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    d_decomp_block0, d_decomp_block1, d_matrix, namps_per_block);                
            }
            cudaStreamSynchronize(stream);
            auto te_1q_out = std::chrono::steady_clock::now();
            auto t_1q_out = std::chrono::duration_cast<std::chrono::nanoseconds>(te_1q_out - ts_1q_out);
            total_1q_out_time += t_1q_out.count() / 1e9;
        }
    }
    auto te_1q = std::chrono::steady_clock::now();
    auto t_1q = std::chrono::duration_cast<std::chrono::nanoseconds>(te_1q - ts_1q);
    total_1q_time += t_1q.count() / 1e9;
    // cudaFree(d_matrix);
}

// Case 1: 両方のqubitがブロック内の場合
__global__ void process_2qubit_within_block_kernel(
    cuDoubleComplex* d_decomp_block, cuDoubleComplex* d_matrix, UINT namps_per_block,
    ITYPE mask_low, ITYPE mask_mid, ITYPE mask_high, ITYPE mask0, ITYPE mask1)
{
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_index < namps_per_block / 4) {
        ITYPE amp_offset0 = (state_index & mask_low) + ((state_index & mask_mid) << 1) 
            + ((state_index & mask_high) << 2);
        ITYPE amp_offset1 = amp_offset0 + mask0;
        ITYPE amp_offset2 = amp_offset0 + mask1;
        ITYPE amp_offset3 = amp_offset1 + mask1;

        cuDoubleComplex a0 = d_decomp_block[amp_offset0];
        cuDoubleComplex a1 = d_decomp_block[amp_offset1];
        cuDoubleComplex a2 = d_decomp_block[amp_offset2];
        cuDoubleComplex a3 = d_decomp_block[amp_offset3];

        // 4x4行列の適用
        d_decomp_block[amp_offset0] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[0], a0), cuCmul(d_matrix[1], a1)), cuCmul(d_matrix[2], a2)), cuCmul(d_matrix[3], a3));
        d_decomp_block[amp_offset1] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[4], a0), cuCmul(d_matrix[5], a1)), cuCmul(d_matrix[6], a2)), cuCmul(d_matrix[7], a3));
        d_decomp_block[amp_offset2] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[8], a0), cuCmul(d_matrix[9], a1)), cuCmul(d_matrix[10], a2)), cuCmul(d_matrix[11], a3));
        d_decomp_block[amp_offset3] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[12], a0), cuCmul(d_matrix[13], a1)), cuCmul(d_matrix[14], a2)), cuCmul(d_matrix[15], a3));
    }
}

// Case 3: 両方のqubitがブロック間（4ブロックにまたがる）
__global__ void process_2qubit_cross_four_blocks_kernel(
    cuDoubleComplex* d_decomp_block0, cuDoubleComplex* d_decomp_block1,
    cuDoubleComplex* d_decomp_block2, cuDoubleComplex* d_decomp_block3,
    cuDoubleComplex* d_matrix, UINT namps_per_block)
{
    ITYPE amp_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (amp_offset < namps_per_block) {    
        cuDoubleComplex a0 = d_decomp_block0[amp_offset];
        cuDoubleComplex a1 = d_decomp_block1[amp_offset];
        cuDoubleComplex a2 = d_decomp_block2[amp_offset];
        cuDoubleComplex a3 = d_decomp_block3[amp_offset];

        cuDoubleComplex new_a0 = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[0], a0), cuCmul(d_matrix[1], a1)), cuCmul(d_matrix[2], a2)), cuCmul(d_matrix[3], a3));
        cuDoubleComplex new_a1 = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[4], a0), cuCmul(d_matrix[5], a1)), cuCmul(d_matrix[6], a2)), cuCmul(d_matrix[7], a3));
        cuDoubleComplex new_a2 = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[8], a0), cuCmul(d_matrix[9], a1)), cuCmul(d_matrix[10], a2)), cuCmul(d_matrix[11], a3));
        cuDoubleComplex new_a3 = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[12], a0), cuCmul(d_matrix[13], a1)), cuCmul(d_matrix[14], a2)), cuCmul(d_matrix[15], a3));

        d_decomp_block0[amp_offset] = new_a0;
        d_decomp_block1[amp_offset] = new_a1;
        d_decomp_block2[amp_offset] = new_a2;
        d_decomp_block3[amp_offset] = new_a3;
    }
}

// Case 2: 片方のqubitがブロック内、片方がブロック間の場合
__global__ void process_2qubit_cross_two_blocks_kernel(
    cuDoubleComplex* d_decomp_block0, cuDoubleComplex* d_decomp_block1, cuDoubleComplex* d_matrix,
    UINT namps_per_block, UINT mask_low, UINT mask_mid, UINT mask0)
{
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_index < namps_per_block / 2) {       
        ITYPE amp_offset0 = (state_index & mask_low) + ((state_index & mask_mid) << 1);
        ITYPE amp_offset1 = amp_offset0 + mask0;
        
        cuDoubleComplex a0 = d_decomp_block0[amp_offset0];
        cuDoubleComplex a1 = d_decomp_block0[amp_offset1];
        cuDoubleComplex a2 = d_decomp_block1[amp_offset0];
        cuDoubleComplex a3 = d_decomp_block1[amp_offset1];

        d_decomp_block0[amp_offset0] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[0], a0), cuCmul(d_matrix[1], a1)), cuCmul(d_matrix[2], a2)), cuCmul(d_matrix[3], a3));
        d_decomp_block0[amp_offset1] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[4], a0), cuCmul(d_matrix[5], a1)), cuCmul(d_matrix[6], a2)), cuCmul(d_matrix[7], a3));
        d_decomp_block1[amp_offset0] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[8], a0), cuCmul(d_matrix[9], a1)), cuCmul(d_matrix[10], a2)), cuCmul(d_matrix[11], a3));
        d_decomp_block1[amp_offset1] = cuCadd(cuCadd(cuCadd(cuCmul(d_matrix[12], a0), cuCmul(d_matrix[13], a1)), cuCmul(d_matrix[14], a2)), cuCmul(d_matrix[15], a3));
    }
}

void CompressedQuantumState::apply_2q_gate_GPU(
    QuantumGateMatrix* gate, 
    const ComplexMatrix& matrix, 
    cuDoubleComplex* d_matrix,
    const std::vector<UINT>& block_ids, 
    void* d_decompressed_SVblocks_void,
    cudaStream_t stream
) {    
    UINT target_index0 = gate->get_target_index_list()[0];
    UINT target_index1 = gate->get_target_index_list()[1];
    const UINT min_qubit_index = get_min_ui(target_index0, target_index1);
    const UINT max_qubit_index = get_max_ui(target_index0, target_index1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE mask_low = min_qubit_mask - 1;
    const ITYPE mask_mid = (max_qubit_mask - 1) ^ mask_low;
    const ITYPE mask_high = ~(max_qubit_mask - 1);
    const ITYPE mask0 = 1ULL << min_qubit_index;
    const ITYPE mask1 = 1ULL << max_qubit_index;
    const ITYPE bmask0 = mask0 >> block_size;
    const ITYPE bmask1 = mask1 >> block_size;

    auto ts_2q = std::chrono::steady_clock::now();
    cuDoubleComplex* d_decomp_blocks = static_cast<cuDoubleComplex*>(d_decompressed_SVblocks_void);
    
    // 行列をGPUにコピー
    auto ts_2q_matrix = std::chrono::steady_clock::now();
    std::vector<std::complex<double>> h_matrix(16);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            h_matrix[i * 4 + j] = matrix(i, j);
        }
    }
    // cuDoubleComplex* d_matrix;
    // cudaMalloc((void**)&d_matrix, 16 * sizeof(cuDoubleComplex));
    cudaMemcpy(d_matrix, h_matrix.data(), 16 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    UINT threadsPerBlock = 256;
    auto te_2q_matrix = std::chrono::steady_clock::now();
    auto t_2q_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(te_2q_matrix - ts_2q_matrix);
    total_2q_matrix_time += t_2q_matrix.count() / 1e9;

    if (max_qubit_index < block_size) {
        // Case 1: 両方のqubitがブロック内
        UINT nelems = namps_per_block / 4;
        UINT blocksPerGrid = (nelems + threadsPerBlock - 1) / threadsPerBlock;

        auto ts_2q_2in = std::chrono::steady_clock::now();
        for (UINT i = 0; i < block_ids.size(); i++) {            
            cuDoubleComplex* d_decomp_block = d_decomp_blocks + i * namps_per_block;
            process_2qubit_within_block_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                d_decomp_block, d_matrix, namps_per_block,
                mask_low, mask_mid, mask_high, mask0, mask1);           
        }
        cudaStreamSynchronize(stream);
        auto te_2q_2in = std::chrono::steady_clock::now();
        auto t_2q_2in = std::chrono::duration_cast<std::chrono::nanoseconds>(te_2q_2in - ts_2q_2in);
        total_2q_2in_time += t_2q_2in.count() / 1e9;
    } else if (min_qubit_index >= block_size) {
        // Case 2: 両方のqubitがブロック間（4ブロックにまたがる）
        std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());
        
        auto ts_2q_2out = std::chrono::steady_clock::now();
        while (target_block_ids.size() > 0) {
            auto ts_2q_2out_search = std::chrono::steady_clock::now();
            UINT block_id0 = target_block_ids.front();
            target_block_ids.pop_front();
            UINT block_id1 = block_id0 + bmask0;
            UINT block_id2 = block_id0 + bmask1;
            UINT block_id3 = block_id1 + bmask1;
            target_block_ids.remove(block_id1);
            target_block_ids.remove(block_id2);
            target_block_ids.remove(block_id3);
            
            UINT local_block_id0 = search_index(block_ids, block_id0);
            UINT local_block_id1 = search_index(block_ids, block_id1);
            UINT local_block_id2 = search_index(block_ids, block_id2);
            UINT local_block_id3 = search_index(block_ids, block_id3);          
            
            // 4ブロックにまたがる場合のカーネル呼び出し
            // ブロック0-1間、ブロック0-2間、ブロック0-3間の振幅を処理
            UINT blocksPerGrid_4block = (namps_per_block + threadsPerBlock - 1) / threadsPerBlock;
            cuDoubleComplex* d_decomp_block0 = d_decomp_blocks + local_block_id0 * namps_per_block;
            cuDoubleComplex* d_decomp_block1 = d_decomp_blocks + local_block_id1 * namps_per_block;
            cuDoubleComplex* d_decomp_block2 = d_decomp_blocks + local_block_id2 * namps_per_block;
            cuDoubleComplex* d_decomp_block3 = d_decomp_blocks + local_block_id3 * namps_per_block;
            auto te_2q_2out_search = std::chrono::steady_clock::now();
            auto t_2q_2out_search = std::chrono::duration_cast<std::chrono::nanoseconds>(te_2q_2out_search - ts_2q_2out_search);
            total_2q_2out_search_time += t_2q_2out_search.count() / 1e9;
            
            // 4x4行列を4ブロックに適用するカーネルを起動
            process_2qubit_cross_four_blocks_kernel<<<blocksPerGrid_4block, threadsPerBlock, 0, stream>>>(
                d_decomp_block0, d_decomp_block1, d_decomp_block2, d_decomp_block3, 
                d_matrix, namps_per_block
            );
        }
        cudaStreamSynchronize(stream);
        auto te_2q_2out = std::chrono::steady_clock::now();
        auto t_2q_2out = std::chrono::duration_cast<std::chrono::nanoseconds>(te_2q_2out - ts_2q_2out);
        total_2q_2out_time += t_2q_2out.count() / 1e9;
    } else {
        // Case 3: 片方のqubitがブロック内、片方がブロック間（2ブロックにまたがる）
        UINT num_elems = namps_per_block / 2;
        UINT blocksPerGrid = (num_elems + threadsPerBlock - 1) / threadsPerBlock;
        std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());

        auto ts_2q_1in1out = std::chrono::steady_clock::now();
        while (target_block_ids.size() > 0) {
            auto ts_2q_1in1out_search = std::chrono::steady_clock::now();
            UINT block_id0 = target_block_ids.front();
            target_block_ids.pop_front();
            UINT block_id1 = block_id0 + bmask1;
            target_block_ids.remove(block_id1);     
            UINT local_block_id0 = search_index(block_ids, block_id0);
            UINT local_block_id1 = search_index(block_ids, block_id1);
            cuDoubleComplex* d_decomp_block0 = d_decomp_blocks + local_block_id0 * namps_per_block;
            cuDoubleComplex* d_decomp_block1 = d_decomp_blocks + local_block_id1 * namps_per_block;
            auto te_2q_1in1out_search = std::chrono::steady_clock::now();
            auto t_2q_1in1out_search = std::chrono::duration_cast<std::chrono::nanoseconds>(te_2q_1in1out_search - ts_2q_1in1out_search);
            total_2q_1in1out_search_time += t_2q_1in1out_search.count() / 1e9;
            
            process_2qubit_cross_two_blocks_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                d_decomp_block0, d_decomp_block1, d_matrix,
                namps_per_block, mask_low, mask_mid, mask0);
        }
        cudaStreamSynchronize(stream);
        auto te_2q_1in1out = std::chrono::steady_clock::now();
        auto t_2q_1in1out = std::chrono::duration_cast<std::chrono::nanoseconds>(te_2q_1in1out - ts_2q_1in1out);
        total_2q_1in1out_time += t_2q_1in1out.count() / 1e9;
    }

    // cudaFree(d_matrix);
    cudaStreamSynchronize(stream);
    auto te_2q = std::chrono::steady_clock::now();
    auto t_2q = std::chrono::duration_cast<std::chrono::nanoseconds>(te_2q - ts_2q);
    total_2q_time += t_2q.count() / 1e9;
}

// CNOT kernel for within-block case (target < block_size)
__global__ void process_cnot_cin_tin(cuDoubleComplex* d_decomp_block, UINT namps_per_block,
    ITYPE low_mask, ITYPE mid_mask, ITYPE high_mask, ITYPE control_mask, ITYPE target_mask)
{
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_index < namps_per_block / 4) {
        ITYPE amp_offset0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
            ((state_index & high_mask) << 2) + control_mask;
        ITYPE amp_offset1 = amp_offset0 | target_mask;
        cuDoubleComplex temp = d_decomp_block[amp_offset0];
        d_decomp_block[amp_offset0] = d_decomp_block[amp_offset1];
        d_decomp_block[amp_offset1] = temp;
    }
}

__global__ void process_cnot_cout_tin(cuDoubleComplex* d_decomp_block, UINT namps_per_block,
    ITYPE low_mask, ITYPE high_mask, ITYPE target_mask)
{
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_index < namps_per_block / 2) {
        ITYPE amp_offset0 = (state_index & low_mask) + ((state_index & high_mask) << 1);
        ITYPE amp_offset1 = amp_offset0 | target_mask;
        cuDoubleComplex temp = d_decomp_block[amp_offset0];
        d_decomp_block[amp_offset0] = d_decomp_block[amp_offset1];
        d_decomp_block[amp_offset1] = temp;
    }
}

// CNOT kernel for cross-block case (target >= block_size, control < block_size)
__global__ void process_cnot_cin_tout(
    cuDoubleComplex* d_decomp_block0, cuDoubleComplex* d_decomp_block1, UINT namps_per_block,
    ITYPE low_mask, ITYPE high_mask, ITYPE control_mask)
{
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_index < namps_per_block / 2) {
        ITYPE amp_offset = (state_index & low_mask) + ((state_index & high_mask) << 1) + control_mask;
        cuDoubleComplex temp = d_decomp_block0[amp_offset];
        d_decomp_block0[amp_offset] = d_decomp_block1[amp_offset];
        d_decomp_block1[amp_offset] = temp;
    }
}

// CNOT kernel for cross-block case (both target and control >= block_size)
__global__ void process_cnot_cout_tout(
    cuDoubleComplex* d_decomp_block0, cuDoubleComplex* d_decomp_block1, UINT namps_per_block)
{
    ITYPE amp_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (amp_offset < namps_per_block) {
        cuDoubleComplex temp = d_decomp_block0[amp_offset];
        d_decomp_block0[amp_offset] = d_decomp_block1[amp_offset];
        d_decomp_block1[amp_offset] = temp;
    }
}

void CompressedQuantumState::apply_cnot_GPU(
    QuantumGateMatrix* gate, 
    const std::vector<UINT>& block_ids, 
    void* decomp_block_pointers_void,
    cudaStream_t stream
) {    
    UINT target_qubit_index = gate->get_target_index_list()[0];
    UINT control_qubit_index = gate->get_control_index_list()[0];
    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;
    const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);
    const ITYPE target_bmask = target_mask >> block_size;
    const ITYPE control_bmask = control_mask >> block_size;

    cuDoubleComplex* d_decompressed_SVblocks = static_cast<cuDoubleComplex*>(decomp_block_pointers_void);
    UINT threadsPerBlock = 256;

#ifdef _USE_MPI
    MPIutil& mpiutil = MPIutil::get_inst();
    UINT rank = mpiutil.get_rank();
#endif

    if (target_qubit_index < block_size) {
        if (control_qubit_index < block_size) {
            UINT nelems = namps_per_block / 4;
            UINT blocksPerGrid = (nelems + threadsPerBlock - 1) / threadsPerBlock;

            for (UINT i = 0; i < block_ids.size(); i++) {
                cuDoubleComplex* d_decomp_block = 
                    d_decompressed_SVblocks + (size_t)i * namps_per_block;
                process_cnot_cin_tin<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    d_decomp_block, namps_per_block,
                    low_mask, mid_mask, high_mask, control_mask, target_mask
                );                
            }            
        } else { // control_qubit_index >= block_size
            const ITYPE _low_mask = target_mask - 1;
            const ITYPE _high_mask = ~_low_mask;
            UINT nelems = namps_per_block / 2;
            UINT blocksPerGrid = (nelems + threadsPerBlock - 1) / threadsPerBlock;

            for (UINT i = 0; i < block_ids.size(); i++) {
#ifdef _USE_MPI
                UINT global_block_id = block_ids[i] + rank * nblocks;
                if ((global_block_id & control_bmask) == 0) continue;
#else                    
                if ((block_ids[i] & control_bmask) == 0) continue;
#endif
                cuDoubleComplex* d_decomp_block = 
                    d_decompressed_SVblocks + (size_t)i * namps_per_block;
                process_cnot_cout_tin<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    d_decomp_block, namps_per_block, _low_mask, _high_mask, target_mask);
            }
        }
    } else { // target_qubit_index >= block_size
        if (control_qubit_index < block_size) {
            ITYPE _low_mask = control_mask - 1;
            ITYPE _high_mask = ~_low_mask;
            UINT nelems = namps_per_block / 2;
            UINT blocksPerGrid = (nelems + threadsPerBlock - 1) / threadsPerBlock;

            std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());
            while (target_block_ids.size() > 0) {
                UINT block_id0 = target_block_ids.front();
                target_block_ids.pop_front();
                UINT block_id1 = block_id0 + target_bmask;
                target_block_ids.remove(block_id1);                
                UINT local_block_id0 = search_index(block_ids, block_id0);
                UINT local_block_id1 = search_index(block_ids, block_id1);
                cuDoubleComplex* d_decomp_block0 = 
                    d_decompressed_SVblocks + (size_t)local_block_id0 * namps_per_block;
                cuDoubleComplex* d_decomp_block1 = 
                    d_decompressed_SVblocks + (size_t)local_block_id1 * namps_per_block;

                process_cnot_cin_tout<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    d_decomp_block0, d_decomp_block1, namps_per_block, 
                    _low_mask, _high_mask, control_mask);                
            }
        } else { // control_qubit_index >= block_size
            UINT blocksPerGrid = (namps_per_block + threadsPerBlock - 1) / threadsPerBlock;

            std::list<UINT> target_block_ids;
            for (UINT i = 0; i < block_ids.size(); i++) {
#ifdef _USE_MPI
                UINT global_block_id = block_ids[i] + rank * nblocks;
                if ((global_block_id & control_bmask) == 0) continue;
#else                    
                if ((block_ids[i] & control_bmask) == 0) continue;
#endif                   
                target_block_ids.push_back(block_ids[i]);
            }

            while (target_block_ids.size() > 0) {
                UINT block_id0 = target_block_ids.front();
                target_block_ids.pop_front();
                UINT block_id1 = block_id0 + target_bmask;
                target_block_ids.remove(block_id1);
                UINT local_block_id0 = search_index(block_ids, block_id0);
                UINT local_block_id1 = search_index(block_ids, block_id1);
                cuDoubleComplex* d_decomp_block0 = 
                    d_decompressed_SVblocks + (size_t)local_block_id0 * namps_per_block;
                cuDoubleComplex* d_decomp_block1 = 
                    d_decompressed_SVblocks + (size_t)local_block_id1 * namps_per_block;
                            
                process_cnot_cout_tout<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    d_decomp_block0, d_decomp_block1, namps_per_block);
            }
        }
    }
}