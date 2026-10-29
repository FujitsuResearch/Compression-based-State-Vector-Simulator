#pragma once

#include "state.hpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

class CompressedQuantumState : public QuantumStateCpu {
private:
    CPPCTYPE* _state_vector;
    Random random;
    // variables for block management
    UINT block_size;
    UINT namps_per_block;
    UINT nblocks;
    // pointers for compressed blocks and their sizes
    std::vector<char*> block_pointers;
    std::vector<size_t> blocks_bytes;
    std::vector<size_t> global_blocks_bytes;
    std::vector<ITYPE> total_SV_bytes;
    // variables for circuit stage management
    std::vector<QuantumGateMatrix*> gate_list;
    std::vector<std::vector<UINT>> circuit_stages;
    std::vector<std::vector<UINT>> block_to_group_list;
    std::vector<UINT> stage_num_groups;
    // variables for (de)compression
    SZ3::Config SZ3_compress_config;
    SZ3::Config SZ3_decompress_config;
    double all_zero_threshold;
#ifdef _USE_GPU_COMP    
    bitcompHandle_t bitcomp_plan{};
    double bitcomp_eps_abs;
    double bitcomp_eps_rel;
    // Todo: prepare multiple matrices for async execution
    static cuDoubleComplex* d_1q_matrix;
    static cuDoubleComplex* d_2q_matrix;
#ifdef _USE_RMM
    // RMM pool memory resource
    static std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_mr;
#endif
#endif

    static double elapsed_seconds(
        const std::chrono::steady_clock::time_point& start,
        const std::chrono::steady_clock::time_point& end) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() / 1e9;
    }

public:
    // timers (seconds)
    double total_nzgrp_time;
    double total_size_time;
    double total_decomp_time;
    double total_matrix_malloc_time;
    double total_gate_time;
    double total_matrix_free_time;
    double total_comp_time;
    double total_zero_check_time;
    double total_hmalloc_time;
    double total_dmalloc_time;
    double total_h2d_time;
    double total_d2h_time;
    double total_dfree_time;
    double total_allgather_time;
    double total_sum_time;
    double total_1q_time;
    double total_1q_matrix_time;
    double total_1q_in_time;
    double total_1q_out_time;
    double total_1q_out_search_time;
    double total_2q_time;
    double total_2q_matrix_time;
    double total_2q_2in_time;
    double total_2q_2out_time;
    double total_2q_2out_search_time;
    double total_2q_1in1out_time;
    double total_2q_1in1out_search_time;

    /**
     * \~japanese-en コンストラクタ
     *
     * @param qubit_count_ 量子ビット数
     * @param block_size_ ブロックサイズ
     */
    explicit CompressedQuantumState(UINT qubit_count_, UINT block_size_)
        : QuantumStateCpu(qubit_count_, true, true) {
        block_size = block_size_;
        namps_per_block = 1U << block_size;
        nblocks = this->_dim / namps_per_block;
        total_nzgrp_time = 0.0;
        total_size_time = 0.0;
        total_decomp_time = 0.0;
        total_matrix_malloc_time = 0.0;
        total_gate_time = 0.0;
        total_matrix_free_time = 0.0;
        total_comp_time = 0.0;
        total_zero_check_time = 0.0;
        total_hmalloc_time = 0.0;
        total_dmalloc_time = 0.0;
        total_h2d_time = 0.0;
        total_d2h_time = 0.0;
        total_dfree_time = 0.0;
        total_allgather_time = 0.0;
        total_sum_time = 0.0;
        total_1q_time = 0.0;
        total_1q_matrix_time = 0.0;
        total_1q_in_time = 0.0;
        total_1q_out_time = 0.0;
        total_1q_out_search_time = 0.0;
        total_2q_time = 0.0;
        total_2q_matrix_time = 0.0;
        total_2q_2in_time = 0.0;
        total_2q_2out_time = 0.0;
        total_2q_2out_search_time = 0.0;
        total_2q_1in1out_time = 0.0;
        total_2q_1in1out_search_time = 0.0;
#ifdef _USE_GPU_COMP
        // bitcomp setting        
        bitcompCreatePlan(
            &bitcomp_plan,
            namps_per_block * 2 * sizeof(double),  // 入力サイズ（bytes）
            BITCOMP_FP64_DATA,                     // データ型：double64
            BITCOMP_LOSSY_FP_TO_SIGNED,            // loss y (fp→signed) モード
            BITCOMP_DEFAULT_ALGO);                // アルゴリズム（適宜
        
        const char* env_bitcomp_rel_eb = std::getenv("BITCOMP_REL_EB");
        if (env_bitcomp_rel_eb != nullptr) {
            bitcomp_eps_rel = std::atof(env_bitcomp_rel_eb);
        } else {
            bitcomp_eps_rel = 1e-3;
        }
        const char* env_bitcomp_abs_eb = std::getenv("BITCOMP_ABS_EB");
        if (env_bitcomp_abs_eb != nullptr) {
            bitcomp_eps_abs = std::atof(env_bitcomp_abs_eb);
        } else {
            bitcomp_eps_abs = 1e-7;
        }

#ifdef _USE_RMM
        // Initialize RMM pool memory resource
        auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t rmm_pool_size_unit = 256;
        size_t pool_size = static_cast<size_t>(free_mem * 0.9);
        pool_size = (pool_size + rmm_pool_size_unit - 1) / rmm_pool_size_unit * rmm_pool_size_unit;

        pool_mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            cuda_mr.get(), pool_size);
        rmm::mr::set_current_device_resource(pool_mr.get());

        auto mr = rmm::mr::get_current_device_resource();
        d_1q_matrix = static_cast<cuDoubleComplex*>(mr->allocate(4 * sizeof(cuDoubleComplex)));
        d_2q_matrix = static_cast<cuDoubleComplex*>(mr->allocate(16 * sizeof(cuDoubleComplex)));
#else
        cudaMalloc((void**)&d_1q_matrix, 4 * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_2q_matrix, 16 * sizeof(cuDoubleComplex));
#endif
#else
        // SZ3 setting
        SZ3::Config config(namps_per_block * 2); // for complex numbers
        config.openmp = true;
        config.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;

        const char* env_sz3_eb_mode = std::getenv("SZ3_EB_MODE");
        const bool use_relative_error =
            env_sz3_eb_mode != nullptr && std::strcmp(env_sz3_eb_mode, "rel") == 0;
        const char* env_sz3_eb_val = std::getenv("SZ3_EB_VAL");
        const double error_bound =
            env_sz3_eb_val != nullptr ? std::atof(env_sz3_eb_val) : 1E-7;

        if (use_relative_error) {
            config.errorBoundMode = SZ3::EB_REL;
            config.relErrorBound = error_bound;
        } else {
            config.errorBoundMode = SZ3::EB_ABS;
            config.absErrorBound = error_bound;
        }
        SZ3_compress_config = config;
        SZ3_decompress_config = config;

        const char* env_all_zero_threshold = std::getenv("ALL_ZERO_THRESHOLD");
        if (env_all_zero_threshold != nullptr) {
            all_zero_threshold = std::atof(env_all_zero_threshold);
        } else {
            all_zero_threshold = error_bound * 10.0;
        }
#endif

#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        UINT rank = mpiutil.get_rank();
#else
        UINT rank = 0;
#endif

        if (rank == 0) {
#ifdef _USE_GPU_COMP
            std::cout << "GPU-based Bitcomp (de)compression" << std::endl;
            std::cout << "  relative error bound: " << bitcomp_eps_rel << std::endl;
            std::cout << "  absolute error bound: " << bitcomp_eps_abs << std::endl;
#ifdef _USE_RMM
            std::cout << "RMM is used for GPU memory pool" << std::endl;
#endif
#else
            if (config.openmp) {
                std::cout << "CPU-based SZ3 (de)compression with OpenMP" << std::endl;
            } else {
                std::cout << "CPU-based SZ3 (de)compression without OpenMP" << std::endl;
            }
            if (use_relative_error) {
                std::cout << "  relative error bound: " << SZ3_compress_config.relErrorBound << std::endl;
            } else {
                std::cout << "  absolute error bound: " << SZ3_compress_config.absErrorBound << std::endl;
            }
            std::cout << "  all-zero threshold: " << all_zero_threshold << std::endl;
#endif
        }

        // allocate memory for block management
        block_pointers = std::vector<char*>(nblocks);
        blocks_bytes = std::vector<size_t>(nblocks);
        ITYPE total_global_nblocks = static_cast<ITYPE>(nblocks) << outer_qc;
        global_blocks_bytes = std::vector<size_t>(total_global_nblocks);
        
#ifdef _USE_MPI
        if (rank == 0) {
            initialize_all_blocks(true);
        } else {
            initialize_all_blocks(false);
        }
#else
        initialize_all_blocks(true);
#endif
          
        _state_vector = nullptr;
    }

    /**
     * \~japanese-en デストラクタ
     */
    virtual ~CompressedQuantumState() {
#ifdef _USE_MPI
        if (this->outer_qc > 0) {
            MPIutil& mpiutil = MPIutil::get_inst();
            mpiutil.release_workarea();
        }
#endif
        for (UINT i = 0; i < nblocks; i++) {
            free(block_pointers[i]);
        }
        if (_state_vector != nullptr) { free(_state_vector); _state_vector = nullptr; }  // SPIKE
#ifdef _USE_GPU_COMP

#ifdef _USE_RMM
        // Clean up RMM allocated memory
        auto mr = rmm::mr::get_current_device_resource();
        if (d_1q_matrix) {
            mr->deallocate(d_1q_matrix, 4 * sizeof(cuDoubleComplex));
            d_1q_matrix = nullptr;
        }
        if (d_2q_matrix) {
            mr->deallocate(d_2q_matrix, 16 * sizeof(cuDoubleComplex));
            d_2q_matrix = nullptr;
        }

        // Reset current device resource first to avoid dangling references
        rmm::mr::set_current_device_resource(nullptr);

        if (pool_mr) {
            pool_mr.release();
        }
#else
        cudaFree(d_1q_matrix);
        cudaFree(d_2q_matrix);
#endif
#endif
// #ifdef _USE_GPU_COMP            
//             cudaFreeHost(block_pointers[i]);
// #else            
//             free(block_pointers[i]);
// #endif
    }
    
    // CPU fallback path (used when GPU compression is disabled)
    char* compress_SV_block(CPPCTYPE* decomp_block, size_t &cmp_size, bool is_all_zero) {
        char* comp_block;
        if (is_all_zero) {
            comp_block = nullptr;
            cmp_size = 0;
        } else {
            comp_block = SZ_compress<double>(SZ3_compress_config, 
                reinterpret_cast<double*>(decomp_block), cmp_size);
        }
        return comp_block;
    }

    // CPU fallback path (used when GPU compression is disabled)
    CPPCTYPE* decompress_SV_block(char* comp_block, size_t cmp_size) { 
        if (comp_block == nullptr) { // all-zero block
            CPPCTYPE* decomp_block = (CPPCTYPE*)malloc(namps_per_block * sizeof(CPPCTYPE));
#ifdef _OPENMP
#pragma omp parallel for
            for (UINT i = 0; i < namps_per_block; i++) {
                decomp_block[i] = 0;
            }
#else            
            memset(decomp_block, 0, namps_per_block * sizeof(CPPCTYPE));
#endif
            return decomp_block;
        } else {
            double* decomp_block = SZ_decompress<double>(SZ3_decompress_config, comp_block, cmp_size);
            return reinterpret_cast<CPPCTYPE*>(decomp_block);
        } 
    }

    void register_SV_block(char* comp_block, size_t comp_size, UINT block_id) {
        if (block_pointers[block_id] != nullptr) {
            free(block_pointers[block_id]);
        }
        blocks_bytes[block_id] = comp_size;
        block_pointers[block_id] = comp_block;
    }

    void initialize_all_blocks(bool with_first_block) {
        size_t cmp_size;
        char* comp_block;
        UINT pos = 0;

        if (with_first_block) {
            // initialize the first block and compress it
            CPPCTYPE* tmp_block = (CPPCTYPE*)malloc(sizeof(CPPCTYPE) * namps_per_block);
            initialize_quantum_state(tmp_block, namps_per_block);
#ifdef _USE_GPU_COMP
            comp_block = compress_SV_block_GPU(tmp_block, cmp_size);
#else
            comp_block = compress_SV_block(tmp_block, cmp_size, false);
#endif
            register_SV_block(comp_block, cmp_size, pos++);
            free(tmp_block);
        }
        
        // initialize the other blocks as all-zero blocks represented by nullptr
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (UINT i = pos; i < nblocks; i++) {
            register_SV_block(nullptr, 0UL, i);
        }    
    }    
    
    void print_info() {
        std::cout << " block_size: " << block_size << std::endl;
        std::cout << " namps_per_block: " << namps_per_block << std::endl;
        std::cout << " nblocks: " << nblocks << std::endl;
    }

    std::string format_size(ITYPE size) {
        char buf[32];
        if (size > (1ULL << 30)) {
            sprintf(buf, "%.3f GiB", (double)size / (1ULL << 30));
        } else if (size > (1ULL << 20)) {
            sprintf(buf, "%.3f MiB", (double)size / (1ULL << 20));
        } else if (size > (1ULL << 10)) {
            sprintf(buf, "%.3f KiB", (double)size / (1ULL << 10));
        } else {
            sprintf(buf, "%llu bytes", size);
        }
        return std::string(buf);
    }

    void append_total_SV_bytes() {
        ITYPE total_bytes = 0ULL;
        ITYPE rank_bytes = 0ULL;

#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        UINT rank = mpiutil.get_rank();

        std::string hostname = "";
        char hostname_buf[256];
        if (gethostname(hostname_buf, sizeof(hostname_buf)) == 0) {
            hostname = std::string(hostname_buf);
        }
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::string time_str = std::ctime(&time_t);
        time_str.pop_back();  // Remove trailing newline
        
        auto ts_allgather = std::chrono::steady_clock::now();
        mpiutil.m_UL_allgather(blocks_bytes.data(), global_blocks_bytes.data(), nblocks);
        auto te_allgather = std::chrono::steady_clock::now();
        total_allgather_time += elapsed_seconds(ts_allgather, te_allgather);

        auto ts_sum = std::chrono::steady_clock::now();
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_bytes)
#endif
        for (UINT i = 0; i < global_blocks_bytes.size(); i++) {
            total_bytes += static_cast<ITYPE>(global_blocks_bytes[i]);
        }

#ifdef _OPENMP
        #pragma omp parallel for reduction(+:rank_bytes)
#endif
        for (UINT i = 0; i < blocks_bytes.size(); i++) {
            rank_bytes += static_cast<ITYPE>(blocks_bytes[i]);
        }
        auto te_sum = std::chrono::steady_clock::now();
        total_sum_time += elapsed_seconds(ts_sum, te_sum);

        if (rank == 0) {
            // for (UINT i = 0; i < nblocks << outer_qc; i++) {
            //     std::cout << "block" << i << ":" << global_blocks_bytes[i] << " ";
            // }
            // std::cout << std::endl;
            std::cout << "Total SV bytes: " << format_size(total_bytes) << std::endl;            
        }       
        std::cout << " Rank " << rank << " SV bytes: " << format_size(rank_bytes) 
              << " (Host: " << hostname << ", Time: " << time_str << ")" << std::endl;
#else
        // Non-MPI: global_blocks_bytes mirrors blocks_bytes (single rank).
        // build_stage_group_execution_plan() reads global_blocks_bytes to
        // detect non-zero blocks, so keep it in sync here as well.
        global_blocks_bytes = blocks_bytes;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_bytes)
#endif
        for (UINT i = 0; i < blocks_bytes.size(); i++) {
            total_bytes += static_cast<ITYPE>(blocks_bytes[i]);
        }
        std::cout << "Total SV bytes: " << format_size(total_bytes) << std::endl;
#endif
        total_SV_bytes.push_back(total_bytes);
    }

    ITYPE get_max_SV_bytes() {
        ITYPE size = 0;
        if (!total_SV_bytes.empty()) {
            size = *std::max_element(total_SV_bytes.begin(), total_SV_bytes.end());
        }
        return size;
    }

    void print_decompressed_state_vector() {
        CPPCTYPE* decomp_block;
        for (UINT i = 0; i < nblocks; i++) {
#ifdef _USE_GPU_COMP
            decomp_block = decompress_SV_block_GPU(block_pointers[i], blocks_bytes[i]);
#else            
            decomp_block = decompress_SV_block(block_pointers[i], blocks_bytes[i]);
#endif
            for (UINT j = 0; j < namps_per_block; j++) {
                std::cout << decomp_block[j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // can be called from Python API
    void register_circuit_stages(
        std::vector<QuantumGateMatrix*> gate_list_, 
        std::vector<std::vector<UINT>> stages_, 
        std::vector<std::vector<UINT>> block_to_group_list_,
        std::vector<UINT> stage_num_groups_
    ) {
        gate_list = gate_list_;
        circuit_stages = stages_;
        block_to_group_list = block_to_group_list_;
        stage_num_groups = stage_num_groups_;
    }

    // backward-compatible interface for external callers
    void register_circuit_stages(
        std::vector<QuantumGateMatrix*> gate_list_,
        std::vector<std::vector<UINT>> stages_,
        std::vector<std::vector<std::vector<UINT>>> block_groups_list_) {
        std::vector<std::vector<UINT>> block_to_group_list_;
        std::vector<UINT> stage_num_groups_;

        block_to_group_list_.reserve(block_groups_list_.size());
        stage_num_groups_.reserve(block_groups_list_.size());

        for (const auto& stage_groups : block_groups_list_) {
            UINT num_groups = (UINT)stage_groups.size();
            stage_num_groups_.push_back(num_groups);

            UINT global_nblocks = 0;
            for (const auto& group : stage_groups) {
                global_nblocks += (UINT)group.size();
            }

            std::vector<UINT> stage_block_to_group(global_nblocks, 0);
            for (UINT group_id = 0; group_id < num_groups; ++group_id) {
                for (auto block_id : stage_groups[group_id]) {
                    stage_block_to_group[block_id] = group_id;
                }
            }
            block_to_group_list_.push_back(std::move(stage_block_to_group));
        }

        register_circuit_stages(
            gate_list_, stages_, block_to_group_list_, stage_num_groups_);
    }

    std::vector<std::vector<UINT>> partition_circuit(
        std::vector<QuantumGateMatrix*> gate_list,
        UINT inner_size
    ) {
        std::vector<std::vector<UINT>> stages; //二次元ベクター(UINT = typedef of unsigned int)
        std::vector<UINT> current_stage;
        std::vector<UINT> global_indices;
        std::vector<UINT> global_targets; // 検証対象の1ゲートがターゲットとするqubitを入れる
        UINT threshold = std::max(inner_size, 2U); // ステージに含むグローバルindices数の閾値

        for(UINT i = 0; i < gate_list.size(); i++) { // investigate all gates one by one
            QuantumGateBase* gate = gate_list[i];
                                    
            // Check for FusedSWAP gate (special handling)
            std::string gate_name = gate->get_name();
            if (gate_name == "FusedSWAP") {
                // partition the current stage
                if (!current_stage.empty()) {
                    stages.push_back(current_stage);                    
                }
                
                // one FusedSWAP gate as a separate stage
                std::vector<UINT> fusedswap_stage = {i};
                stages.push_back(fusedswap_stage);
                
                current_stage.clear();
                global_indices.clear();
                continue;
            }
            
            // find target global indices (both targets and controls)            
            auto target_list = gate->get_target_index_list();
            for (auto target : target_list) {
                if (target >= block_size) {
                    global_targets.push_back(target);
                }
            }
            auto control_list = gate->get_control_index_list();
            for (auto control : control_list) {
                if (control >= block_size) {
                    global_targets.push_back(control);
                }
            }
            
            std::vector<UINT> tmp_global_indices = global_indices; // copy
            for (auto gt : global_targets) {
                if (std::find(tmp_global_indices.begin(), tmp_global_indices.end(), gt) == tmp_global_indices.end()) {
                    tmp_global_indices.push_back(gt);
                }
            }

            if (tmp_global_indices.size() > threshold) {               
                // partition the current stage
                stages.push_back(current_stage);                
                                
                current_stage.clear();
                tmp_global_indices = global_targets;  // Use global_targets directly
                global_indices.clear();
                global_targets.clear();
            }
            
            // add the new global indices
            global_indices = tmp_global_indices;
            global_targets.clear();
            
            // add the gate to the current stage
            current_stage.push_back(i);
        }
        
        if (!current_stage.empty()) {
            stages.push_back(current_stage);           
        }
        
        return stages;
    }

    std::vector<std::vector<UINT>> make_block_groups(
        std::vector<QuantumGateMatrix*> gate_list,
        const std::vector<std::vector<UINT>>& stages
    ) {    
        std::vector<std::vector<UINT>> _block_to_group_list;
        std::vector<UINT> _stage_num_groups;
        UINT nqubits = this->_qubit_count;
        
        for (const auto& stage : stages) {
            std::set<UINT> local_inner_indices;
            
            // Initialize with SV_block_size indices (0 to SV_block_size-1)
            for (UINT i = 0; i < block_size; i++) {
                local_inner_indices.insert(i);
            }
            
            // Find inner indices used in this stage
            for (auto gate_index : stage) {
                QuantumGateBase* gate = gate_list[gate_index];
                
                auto target_list = gate->get_target_index_list();
                for (auto target : target_list) {
                    if (target >= block_size) {
                        local_inner_indices.insert(target);
                    }
                }
                
                auto control_list = gate->get_control_index_list();
                for (auto control : control_list) {
                    if (control >= block_size) {
                        local_inner_indices.insert(control);
                    }
                }
            }
            
            // Determine outer indices
            std::vector<UINT> outer_indices;
            for (UINT i = 0; i < nqubits; i++) {
                if (local_inner_indices.find(i) == local_inner_indices.end()) {
                    outer_indices.push_back(i);
                }
            }
            
            // Make block groups based on outer indices
            ITYPE global_nblocks = 1ULL << (nqubits - block_size); // 2^(nqubits - block_size)
#ifdef _USE_MPI
            MPIutil& mpiutil = MPIutil::get_inst();
            UINT rank = mpiutil.get_rank();
            UINT my_nblocks = 1U << (nqubits - block_size - outer_qc);
            UINT first_block = rank * my_nblocks;
#else
            UINT my_nblocks = static_cast<UINT>(global_nblocks);
            UINT first_block = 0U;
#endif
            std::vector<UINT> my_group_ids(my_nblocks);
            std::vector<UINT> global_group_ids(global_nblocks);
            
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (UINT block_id = first_block; block_id < first_block + my_nblocks; block_id++) {
                UINT group_id = 0;                
                // Extract bits at outer indices positions using direct bit operations
                for (UINT i = 0; i < outer_indices.size(); i++) {
                    UINT oid = outer_indices[i];
                    UINT bit_pos_in_block_id = oid - block_size;  // Bit position within block_id
                    UINT bit_value = (block_id >> bit_pos_in_block_id) & 1;
                    group_id |= (bit_value << i);  // Bit i of group_id corresponds to outer_indices[i]
                }                
                my_group_ids[block_id - first_block] = group_id;
            }
           
#ifdef _USE_MPI
            mpiutil.m_UINT_allgather(my_group_ids.data(), global_group_ids.data(), my_nblocks);
#else
            global_group_ids = my_group_ids;            
#endif           
            
            UINT num_groups = 1U << outer_indices.size(); // 2^len(outer_indices)
            _stage_num_groups.push_back(num_groups);
            _block_to_group_list.push_back(std::move(global_group_ids));
        }

        stage_num_groups = std::move(_stage_num_groups);
        
        return _block_to_group_list;
    }

    void partition_and_register_circuit(
        std::vector<QuantumGateMatrix*> gate_list, 
        UINT inner_size, 
        int verbose = 0
    ) {
        auto stages = partition_circuit(gate_list, inner_size);        
        auto _block_to_group_list = make_block_groups(gate_list, stages);
        auto _stage_num_groups = stage_num_groups;
        register_circuit_stages(
            gate_list, stages, _block_to_group_list, _stage_num_groups);

#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        UINT rank = mpiutil.get_rank();
        if (rank != 0) {
            return;
        }
#endif

        if (verbose > 0) {
            std::cout << "Circuit Stages: [";
            for (size_t stage_idx = 0; stage_idx < stages.size(); stage_idx++) {
                const auto& stage = stages[stage_idx];
                if (stage_idx == stages.size() - 1) {
                    std::cout << stage.size() << "]" << std::endl;
                } else {
                    std::cout << stage.size() << ", ";
                }
            }
        }

        if (verbose > 1) {
            std::cout << "\n=== Block Groups ===" << std::endl;
            for (size_t stage_idx = 0; stage_idx < _block_to_group_list.size(); stage_idx++) {
                std::cout << "Stage " << stage_idx << " block groups:" << std::endl;
                std::vector<std::vector<UINT>> stage_block_groups(
                    _stage_num_groups[stage_idx]);
                for (UINT block_id = 0;
                     block_id < _block_to_group_list[stage_idx].size(); ++block_id) {
                    UINT group_id = _block_to_group_list[stage_idx][block_id];
                    stage_block_groups[group_id].push_back(block_id);
                }
                for (size_t group_idx = 0; group_idx < stage_block_groups.size(); group_idx++) {
                    std::cout << "  Group " << group_idx << ": [";
                    const auto& group = stage_block_groups[group_idx];
                    for (size_t i = 0; i < group.size(); i++) {
                        std::cout << group[i];
                        if (i < group.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }        
    }

    void print_timers() {
#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        UINT rank = mpiutil.get_rank();
#else
        UINT rank = 0;
#endif
        if (rank == 0) {
            std::cout << "\n==== Time [sec] ====" << std::endl;
            std::cout << "  Nzgrp     : " << total_nzgrp_time << " s" << std::endl;
            std::cout << "  Size      : " << total_size_time << " s" << std::endl;
            std::cout << "  Dmalloc   : " << total_dmalloc_time << " s" << std::endl;
            std::cout << "  H2D       : " << total_h2d_time << " s" << std::endl;
            std::cout << "  Decomp    : " << total_decomp_time << " s" << std::endl;
            std::cout << "  Matrix malloc : " << total_matrix_malloc_time << " s" << std::endl;
            std::cout << "  Gate      : " << total_gate_time << " s" << std::endl;
            std::cout << "    1q overall: " << total_1q_time << " s" << std::endl;
            std::cout << "      1q matrix : " << total_1q_matrix_time << " s" << std::endl;
            std::cout << "      1q in     : " << total_1q_in_time << " s" << std::endl;
            std::cout << "      1q out    : " << total_1q_out_time << " s" << std::endl;
            std::cout << "        1q out search: " << total_1q_out_search_time << " s" << std::endl;
            std::cout << "    2q overall: " << total_2q_time << " s" << std::endl;
            std::cout << "      2q matrix : " << total_2q_matrix_time << " s" << std::endl;
            std::cout << "      2q 2in    : " << total_2q_2in_time << " s" << std::endl;
            std::cout << "      2q 2out   : " << total_2q_2out_time << " s" << std::endl;
            std::cout << "        2q 2out search: " << total_2q_2out_search_time << " s" << std::endl;
            std::cout << "      2q 1in1out: " << total_2q_1in1out_time << " s" << std::endl;
            std::cout << "        2q 1in1out search: " << total_2q_1in1out_search_time << " s" << std::endl;
            std::cout << "  Matrix free: " << total_matrix_free_time << " s" << std::endl;
            std::cout << "  Comp      : " << total_comp_time << " s" << std::endl;
            std::cout << "    Zero check: " << total_zero_check_time << " s" << std::endl;
            std::cout << "  Hmalloc   : " << total_hmalloc_time << " s" << std::endl;
            std::cout << "  D2H       : " << total_d2h_time << " s" << std::endl;
            std::cout << "  Dfree     : " << total_dfree_time << " s" << std::endl;
            std::cout << "  Fused swap: " << fused_swap_time << " s" << std::endl;
            std::cout << "  Allgather : " << total_allgather_time << " s" << std::endl;
            std::cout << "  Bytes sum : " << total_sum_time << " s" << std::endl;
            double total_time = total_nzgrp_time + total_size_time + total_dmalloc_time +
                                total_h2d_time + total_decomp_time + total_gate_time +
                                total_comp_time + total_hmalloc_time + total_d2h_time +
                                total_dfree_time + fused_swap_time + total_allgather_time +
                                total_sum_time + total_matrix_malloc_time + total_matrix_free_time;
            std::cout << "Total: " << total_time << " s" << std::endl;
            std::cout << "=====================" << std::endl;
        }
    }

#ifdef _USE_GPU_COMP
    // implemented in "compressed_state_cuda.cu"
    char* compress_SV_block_GPU(CPPCTYPE* decomp_block, size_t &cmp_size);
    CPPCTYPE* decompress_SV_block_GPU(char* comp_block, size_t cmp_size);
    void apply_1q_gate_GPU(QuantumGateMatrix* gate, const ComplexMatrix& matrix, cuDoubleComplex* d_matrix,
        const std::vector<UINT>& block_ids, void *d_decompressed_SVblocks, cudaStream_t stream);
    void apply_2q_gate_GPU(QuantumGateMatrix* gate, const ComplexMatrix& matrix, cuDoubleComplex* d_matrix,
        const std::vector<UINT>& block_ids, void *d_decompressed_SVblocks, cudaStream_t stream);
    void apply_cnot_GPU(QuantumGateMatrix* gate, const std::vector<UINT>& block_ids, 
        void *d_decompressed_SVblocks, cudaStream_t stream);
    void process_SV_group_GPU(const std::vector<UINT>& gate_ids, const std::vector<UINT>& block_group, 
        int PU_id, double& decomp_time, double& gate_time, double& comp_time, 
        double& hmalloc_time, double& dmalloc_time, double& h2d_time, double& d2h_time, 
        double& free_time, double& size_time);
#endif
    
    struct StageGroupExecutionPlan {
        std::vector<UINT> active_group_ids;
        std::vector<UINT> group_block_offsets;
        std::vector<UINT> grouped_stage_blocks;
    };

    StageGroupExecutionPlan build_stage_group_execution_plan(
        const std::vector<UINT>& stage_block_to_group, UINT num_groups) const {
        StageGroupExecutionPlan plan;

        std::vector<UINT> non_zero_block_counts_per_group(num_groups, 0);
        for (UINT block_id = 0; block_id < stage_block_to_group.size(); ++block_id) {
            if (global_blocks_bytes[block_id] != 0) {
                UINT group_id = stage_block_to_group[block_id];
                non_zero_block_counts_per_group[group_id]++;
            }
        }

        plan.active_group_ids.reserve(num_groups);
        for (UINT group_id = 0; group_id < num_groups; ++group_id) {
            if (non_zero_block_counts_per_group[group_id] > 0) {
                plan.active_group_ids.push_back(group_id);
            }
        }

        std::vector<UINT> total_block_counts_per_group(num_groups, 0);
        for (UINT block_id = 0; block_id < stage_block_to_group.size(); ++block_id) {
            UINT group_id = stage_block_to_group[block_id];
            total_block_counts_per_group[group_id]++;
        }

        plan.group_block_offsets.assign(num_groups + 1, 0);
        for (UINT group_id = 0; group_id < num_groups; ++group_id) {
            plan.group_block_offsets[group_id + 1] =
                plan.group_block_offsets[group_id] +
                total_block_counts_per_group[group_id];
        }

        plan.grouped_stage_blocks.resize(plan.group_block_offsets[num_groups]);
        std::vector<UINT> next_group_write_offset = plan.group_block_offsets;
        for (UINT block_id = 0; block_id < stage_block_to_group.size(); ++block_id) {
            UINT group_id = stage_block_to_group[block_id];
            plan.grouped_stage_blocks[next_group_write_offset[group_id]++] = block_id;
        }

        return plan;
    }

    // apply the registered quantum circuit stage by stage (group by group) 
    void apply_circuit() {
#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        int rank = mpiutil.get_rank();
        int local_rank = mpiutil.get_localrank();
        int local_size = mpiutil.get_localsize();
#else
        int rank = 0;
        int local_rank = 0;
        int local_size = 1;
#endif
            
#ifdef _USE_GPU_COMP
        int my_num_PUs = 0;
        const char* env_num_gpus = std::getenv("QOMPSIM_GPUS");
        if (env_num_gpus != nullptr) {
            my_num_PUs = std::atoi(env_num_gpus);
        } else {
            cudaGetDeviceCount(&my_num_PUs);
        }
        my_num_PUs = std::max(1, my_num_PUs / local_size);
#else
        int my_num_PUs = 2;
        const char* env_cpu_parts = std::getenv("QOMPSIM_CPU_PARTITIONS");
        if (env_cpu_parts != nullptr) {
            my_num_PUs = std::atoi(env_cpu_parts);
        }
#endif

        int num_streams = 1;
        const char* env_num_streams = std::getenv("QOMPSIM_STREAMS_PER_PU");
        if (env_num_streams != nullptr) {
            num_streams = std::atoi(env_num_streams);
        }

        if (rank == 0) {
            std::cout << "# of PUs per rank: " << my_num_PUs << std::endl;
            std::cout << "# of streams per PU: " << num_streams << std::endl;
        }

        append_total_SV_bytes();
        for (UINT stage_id = 0; stage_id < circuit_stages.size(); stage_id++) {
#ifdef _USE_MPI 
            auto first_gate = gate_list[circuit_stages[stage_id][0]];
            if (first_gate->get_name() == "FusedSWAP") {
                assert(circuit_stages[stage_id].size() == 1);       

                auto ts = std::chrono::steady_clock::now();
                auto target_index_list = first_gate->get_target_index_list();
                UINT swap_width = target_index_list.size() / 2;
                if (rank == 0)
                    std::cout << "Applying FusedSWAP gate (width=" << swap_width << ")" << std::endl;
                apply_fused_swap(target_index_list[0], target_index_list[swap_width], swap_width);
                auto te = std::chrono::steady_clock::now();
                fused_swap_time += elapsed_seconds(ts, te);
                
                if (rank == 0)
                    std::cout << "[Stage " << stage_id << "/" << circuit_stages.size() - 1 << "] ";
                append_total_SV_bytes();
                continue;
            }
#endif                    

            // make a list of non-zero block group IDs
            auto ts_nrgrp = std::chrono::steady_clock::now();
            const auto& stage_block_to_group = block_to_group_list[stage_id];
            UINT num_groups = stage_num_groups[stage_id];
            StageGroupExecutionPlan execution_plan =
                build_stage_group_execution_plan(stage_block_to_group, num_groups);

            auto te_nrgrp = std::chrono::steady_clock::now();
            total_nzgrp_time += elapsed_seconds(ts_nrgrp, te_nrgrp);

            for (auto gid : execution_plan.active_group_ids) {
                UINT begin = execution_plan.group_block_offsets[gid];
                UINT end = execution_plan.group_block_offsets[gid + 1];
#ifdef _USE_GPU_COMP
                std::vector<UINT> block_group(
                    execution_plan.grouped_stage_blocks.begin() + begin,
                    execution_plan.grouped_stage_blocks.begin() + end);
                process_SV_group_GPU(circuit_stages[stage_id], block_group,
                    local_rank, total_decomp_time, total_gate_time, total_comp_time, 
                    total_hmalloc_time, total_dmalloc_time, total_h2d_time, total_d2h_time, 
                    total_dfree_time, total_size_time);
#else
                process_SV_group(
                    circuit_stages[stage_id], execution_plan.grouped_stage_blocks, begin, end,
                    total_decomp_time, total_gate_time, total_comp_time);
#endif
            }

            if (rank == 0)
                std::cout << "[Stage " << stage_id << "/" << circuit_stages.size() - 1 << "] ";
            append_total_SV_bytes();
        }
        // Timer print for debugging and performance analysis
        // print_timers();
    }

    bool check_all_zero(CPPCTYPE* decomp_block) {
        bool is_all_zero = true;
#ifdef _OPENMP        
#pragma omp parallel for reduction(&& : is_all_zero)
#endif
        for (UINT i = 0; i < namps_per_block; ++i) {
            if (std::abs(decomp_block[i].real()) > all_zero_threshold 
                || std::abs(decomp_block[i].imag()) > all_zero_threshold) {
                is_all_zero = false;
            }
        }
        return is_all_zero;
    }

    void process_SV_group(const std::vector<UINT>& gate_ids,
        const std::vector<UINT>& grouped_stage_blocks, UINT begin,
        UINT end,
        double& decomp_time, double& gate_time, double& comp_time
    ) {
        UINT block_id;
        size_t cmp_size;
        char* comp_block;
        std::vector<UINT> my_block_ids;
   
#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        UINT rank = mpiutil.get_rank();
        for (UINT idx = begin; idx < end; ++idx) {
            UINT global_block_id = grouped_stage_blocks[idx];
            if (global_block_id / nblocks == rank) {
                my_block_ids.push_back(global_block_id % nblocks);
            }
        }
#else
        my_block_ids.assign(
            grouped_stage_blocks.begin() + begin,
            grouped_stage_blocks.begin() + end);
#endif

        auto ts_decomp = std::chrono::steady_clock::now();
        CPPCTYPE* decomp_block_pointers[my_block_ids.size()];
        // decompress the blocks in the group
        for (UINT i = 0; i < my_block_ids.size(); i++) {
            block_id = my_block_ids[i];
            decomp_block_pointers[i] = decompress_SV_block(
                block_pointers[block_id], blocks_bytes[block_id]);
// #ifdef _USE_MPI                
//             std::cout << "rank" << rank << ": Decompressed block " << block_id << std::endl;
// #endif 
        }
        auto te_decomp = std::chrono::steady_clock::now();
        decomp_time += elapsed_seconds(ts_decomp, te_decomp);

        // apply all the gates in the stage to the block group
        auto ts_gate = std::chrono::steady_clock::now();
        for (auto gate_id : gate_ids) {
            auto gate = gate_list[gate_id];
            ComplexMatrix matrix = gate->get_matrix();
            // apply the gate to the decompressed blocks
            if (gate->get_name() == "CNOT") {
                apply_cnot(gate, my_block_ids, decomp_block_pointers);            
            } else if (matrix.size() == 4) {
                apply_1q_gate(gate, matrix, my_block_ids, decomp_block_pointers);
            } else if (matrix.size() == 16) {
                apply_2q_gate(gate, matrix, my_block_ids, decomp_block_pointers);
            } else {
                throw NotImplementedException(
                    "Error: CompressedQuantumState::process_SV_group: "
                    "only single- or two-qubit gates are implemented");
            }
        }
        auto te_gate = std::chrono::steady_clock::now();
        gate_time += elapsed_seconds(ts_gate, te_gate);

        // compress the blocks in the group and register them
        auto ts_comp = std::chrono::steady_clock::now();
        for (UINT i = 0; i < my_block_ids.size(); i++) {
            block_id = my_block_ids[i];
            if (block_pointers[block_id] == nullptr) {
                // check if the block is still all-zero
                comp_block = compress_SV_block(decomp_block_pointers[i], cmp_size, 
                    check_all_zero(decomp_block_pointers[i]));
            } else {
                comp_block = compress_SV_block(decomp_block_pointers[i], cmp_size, false);
            }
            register_SV_block(comp_block, cmp_size, block_id);
            free(decomp_block_pointers[i]);
        }
        auto te_comp = std::chrono::steady_clock::now();
        comp_time += elapsed_seconds(ts_comp, te_comp);
    }

    // search a local index in block_ids for a given block_id
    UINT search_index(const std::vector<UINT>& block_ids, UINT block_id) {
        UINT index;
        bool found = false;
        for (UINT i = 0; i < block_ids.size(); i++) {
            if (block_ids[i] == block_id) {
                index = i;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::out_of_range(
                "Error: CompressedQuantumState::search_index: "
                "block_id not found in block_ids");
        }
        return index;
    }

    std::pair<UINT, UINT> pop_block_pair(
        std::list<UINT>& pending_block_ids, UINT pair_delta) {
        UINT block_id0 = pending_block_ids.front();
        pending_block_ids.pop_front();
        UINT block_id1 = block_id0 + pair_delta;
        pending_block_ids.remove(block_id1);
        return {block_id0, block_id1};
    }

    std::pair<UINT, UINT> find_local_block_pair_indices(
        const std::vector<UINT>& block_ids, UINT block_id0, UINT block_id1) {
        UINT local_block_id0 = search_index(block_ids, block_id0);
        UINT local_block_id1 = search_index(block_ids, block_id1);
        return {local_block_id0, local_block_id1};
    }

    void apply_1q_gate(
        QuantumGateMatrix* gate, 
        const ComplexMatrix& matrix, 
        const std::vector<UINT>& block_ids, 
        CPPCTYPE** decomp_block_pointers
    ) {
        UINT target_qubit_index = gate->get_target_index_list()[0];
        const ITYPE mask = 1ULL << target_qubit_index;
        const ITYPE mask_low = mask - 1;
        const ITYPE mask_high = ~mask_low;
        const ITYPE bmask = mask >> block_size;
        ITYPE state_index;

        if (target_qubit_index < block_size) {
            // apply the gate within each block
            for (UINT i = 0; i < block_ids.size(); i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (state_index = 0; state_index < namps_per_block / 2; state_index++) {
                    // determine the two basis indices affected by the gate
                    ITYPE amp_offset0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
                    ITYPE amp_offset1 = amp_offset0 | mask;
                    CPPCTYPE a0 = decomp_block_pointers[i][amp_offset0];
                    CPPCTYPE a1 = decomp_block_pointers[i][amp_offset1];
                    CPPCTYPE new_a0 = matrix(0,0) * a0 + matrix(0,1) * a1;
                    CPPCTYPE new_a1 = matrix(1,0) * a0 + matrix(1,1) * a1;
                    decomp_block_pointers[i][amp_offset0] = new_a0;
                    decomp_block_pointers[i][amp_offset1] = new_a1;
                }
            }
        } else {
            bool diagonal = false;
            if (matrix(0,1) == 0.0 and matrix(1,0) == 0.0) {
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
                    // update only the diagonal elements
#ifdef _OPENMP
#pragma omp parallel for
#endif                    
                    for (ITYPE amp_offset = 0; amp_offset < namps_per_block; amp_offset += 2) {
                        // update the two amplitudes                       
                        decomp_block_pointers[i][amp_offset] *= matrix(isone, isone);
                        decomp_block_pointers[i][amp_offset + 1] *= matrix(isone, isone);
                    }
                }
            } else {
                // apply the gate between pairs of blocks
                std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());              
                while (target_block_ids.size() > 0) {
                    auto [block_id0, block_id1] =
                        pop_block_pair(target_block_ids, bmask);
                    auto [local_block_id0, local_block_id1] =
                        find_local_block_pair_indices(block_ids, block_id0, block_id1);
#ifdef _OPENMP
#pragma omp parallel for
#endif                                  
                    for (ITYPE amp_offset = 0; amp_offset < namps_per_block; amp_offset++) {
                        // update the two amplitudes         
                        CPPCTYPE a0 = decomp_block_pointers[local_block_id0][amp_offset];
                        CPPCTYPE a1 = decomp_block_pointers[local_block_id1][amp_offset];
                        CPPCTYPE new_a0 = matrix(0,0) * a0 + matrix(0,1) * a1;
                        CPPCTYPE new_a1 = matrix(1,0) * a0 + matrix(1,1) * a1;
                        decomp_block_pointers[local_block_id0][amp_offset] = new_a0;
                        decomp_block_pointers[local_block_id1][amp_offset] = new_a1;
                    }                    
                }
            }
        }
    }

    void apply_cnot(QuantumGateMatrix* gate, const std::vector<UINT>& block_ids, CPPCTYPE** decomp_block_pointers
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
        ITYPE state_index;

#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        UINT rank = mpiutil.get_rank();
        // std::cout << "rank" << rank << ": Applying CNOT gate with control qubit " 
        //           << control_qubit_index << " and target qubit " << target_qubit_index << std::endl;
#endif

        if (target_qubit_index < block_size) {
            // apply the gate within each block
            if (control_qubit_index < block_size) {
                for (UINT i = 0; i < block_ids.size(); i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif                     
                    for (state_index = 0; state_index < namps_per_block / 4; state_index++) {
                        // determine the two basis indices affected by the gate
                        ITYPE amp_offset0 = (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                            ((state_index & high_mask) << 2) + control_mask;
                        ITYPE amp_offset1 = amp_offset0 | target_mask;
                        CPPCTYPE temp = decomp_block_pointers[i][amp_offset0];
                        decomp_block_pointers[i][amp_offset0] = decomp_block_pointers[i][amp_offset1];
                        decomp_block_pointers[i][amp_offset1] = temp;                    
                    }
                }
            } else { // control_qubit_index >= block_size
                const ITYPE _low_mask = target_mask - 1;
                const ITYPE _high_mask = ~_low_mask;
                for (UINT i = 0; i < block_ids.size(); i++) {
#ifdef _USE_MPI                   
                    UINT global_block_id = block_ids[i] + rank * nblocks;
                    if ((global_block_id & control_bmask) == 0) continue;
#else                    
                    if ((block_ids[i] & control_bmask) == 0) continue;
#endif                     
#ifdef _OPENMP
#pragma omp parallel for
#endif                
                    for (state_index = 0; state_index < namps_per_block / 2; state_index++) {
                        // determine the two basis indices affected by the gate
                        ITYPE amp_offset0 = (state_index & _low_mask) + ((state_index & _high_mask) << 1);
                        ITYPE amp_offset1 = amp_offset0 | target_mask;
                        CPPCTYPE temp = decomp_block_pointers[i][amp_offset0];
                        decomp_block_pointers[i][amp_offset0] = decomp_block_pointers[i][amp_offset1];
                        decomp_block_pointers[i][amp_offset1] = temp;
                    }
                }
            }
        } else { // target_qubit_index >= block_size
            // apply the gate between pairs of blocks
            if (control_qubit_index < block_size) {
                ITYPE _low_mask = control_mask - 1;
                ITYPE _high_mask = ~_low_mask;

                std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());
                while (target_block_ids.size() > 0) {
                    auto [block_id0, block_id1] =
                        pop_block_pair(target_block_ids, target_bmask);
                    auto [local_block_id0, local_block_id1] =
                        find_local_block_pair_indices(block_ids, block_id0, block_id1);
#ifdef _OPENMP
#pragma omp parallel for
#endif                    
                    for (ITYPE state_index = 0; state_index < namps_per_block / 2; state_index++) {     
                        ITYPE amp_offset = (state_index & _low_mask) +
                            ((state_index & _high_mask) << 1) + control_mask;
                        // update the two amplitudes
                        CPPCTYPE temp = decomp_block_pointers[local_block_id0][amp_offset];
                        decomp_block_pointers[local_block_id0][amp_offset] = 
                            decomp_block_pointers[local_block_id1][amp_offset];
                        decomp_block_pointers[local_block_id1][amp_offset] = temp;
                    }
                }
            } else { // control_qubit_index >= block_size
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
                    auto [block_id0, block_id1] =
                        pop_block_pair(target_block_ids, target_bmask);
                    auto [local_block_id0, local_block_id1] =
                        find_local_block_pair_indices(block_ids, block_id0, block_id1);
#ifdef _OPENMP
#pragma omp parallel for
#endif                    
                    for (ITYPE amp_offset = 0; amp_offset < namps_per_block; amp_offset++) {
                        // update the two amplitudes
                        CPPCTYPE temp = decomp_block_pointers[local_block_id0][amp_offset];
                        decomp_block_pointers[local_block_id0][amp_offset] = 
                            decomp_block_pointers[local_block_id1][amp_offset];
                        decomp_block_pointers[local_block_id1][amp_offset] = temp;
                    }
                }
            }
        }
    }

    void apply_2q_gate(
        QuantumGateMatrix* gate, 
        const ComplexMatrix& matrix, 
        const std::vector<UINT>& block_ids, 
        CPPCTYPE** decomp_block_pointers
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
        ITYPE state_index;        

       if (max_qubit_index < block_size) {
            // apply the gate within each block
            for (UINT i = 0; i < block_ids.size(); i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif                
                for (state_index = 0; state_index < namps_per_block / 4; state_index++) {
                    // determine the two basis indices affected by the gate
                    ITYPE amp_offset0 = 
                        (state_index & mask_low) + ((state_index & mask_mid) << 1) + ((state_index & mask_high) << 2);
                    ITYPE amp_offset1 = amp_offset0 + mask0;
                    ITYPE amp_offset2 = amp_offset0 + mask1;
                    ITYPE amp_offset3 = amp_offset1 + mask1;

                    CPPCTYPE a0 = decomp_block_pointers[i][amp_offset0];
                    CPPCTYPE a1 = decomp_block_pointers[i][amp_offset1];
                    CPPCTYPE a2 = decomp_block_pointers[i][amp_offset2];
                    CPPCTYPE a3 = decomp_block_pointers[i][amp_offset3];
                    CPPCTYPE new_a0 = matrix(0,0) * a0 + matrix(0,1) * a1 + matrix(0,2) * a2 + matrix(0,3) * a3;
                    CPPCTYPE new_a1 = matrix(1,0) * a0 + matrix(1,1) * a1 + matrix(1,2) * a2 + matrix(1,3) * a3;
                    CPPCTYPE new_a2 = matrix(2,0) * a0 + matrix(2,1) * a1 + matrix(2,2) * a2 + matrix(2,3) * a3;
                    CPPCTYPE new_a3 = matrix(3,0) * a0 + matrix(3,1) * a1 + matrix(3,2) * a2 + matrix(3,3) * a3;
                    decomp_block_pointers[i][amp_offset0] = new_a0;
                    decomp_block_pointers[i][amp_offset1] = new_a1;
                    decomp_block_pointers[i][amp_offset2] = new_a2;
                    decomp_block_pointers[i][amp_offset3] = new_a3;
                }
            }
        } else if (min_qubit_index >= block_size) {
            // apply the gate across four blocks
            std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());
            while (target_block_ids.size() > 0) {
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
#ifdef _OPENMP
#pragma omp parallel for
#endif                
                for (ITYPE amp_offset = 0; amp_offset < namps_per_block; amp_offset++) {
                    // update the four amplitudes
                    CPPCTYPE a0 = decomp_block_pointers[local_block_id0][amp_offset];
                    CPPCTYPE a1 = decomp_block_pointers[local_block_id1][amp_offset];
                    CPPCTYPE a2 = decomp_block_pointers[local_block_id2][amp_offset];
                    CPPCTYPE a3 = decomp_block_pointers[local_block_id3][amp_offset];
                    CPPCTYPE new_a0 = matrix(0,0) * a0 + matrix(0,1) * a1 + matrix(0,2) * a2 + matrix(0,3) * a3;
                    CPPCTYPE new_a1 = matrix(1,0) * a0 + matrix(1,1) * a1 + matrix(1,2) * a2 + matrix(1,3) * a3;
                    CPPCTYPE new_a2 = matrix(2,0) * a0 + matrix(2,1) * a1 + matrix(2,2) * a2 + matrix(2,3) * a3;
                    CPPCTYPE new_a3 = matrix(3,0) * a0 + matrix(3,1) * a1 + matrix(3,2) * a2 + matrix(3,3) * a3;
                    decomp_block_pointers[local_block_id0][amp_offset] = new_a0;
                    decomp_block_pointers[local_block_id1][amp_offset] = new_a1;
                    decomp_block_pointers[local_block_id2][amp_offset] = new_a2;
                    decomp_block_pointers[local_block_id3][amp_offset] = new_a3;
                }
            }
        } else {
            // apply the gate across two blocks
            std::list<UINT> target_block_ids(block_ids.begin(), block_ids.end());
            while (target_block_ids.size() > 0) {
                auto [block_id0, block_id1] =
                    pop_block_pair(target_block_ids, bmask1);
                auto [local_block_id0, local_block_id1] =
                    find_local_block_pair_indices(block_ids, block_id0, block_id1);
#ifdef _OPENMP
#pragma omp parallel for
#endif                
                for (state_index = 0; state_index < namps_per_block / 2; state_index++) {
                    // update the four amplitudes
                    ITYPE amp_offset0 = (state_index & mask_low) + ((state_index & mask_mid) << 1);
                    ITYPE amp_offset1 = amp_offset0 + mask0;
                    CPPCTYPE a0 = decomp_block_pointers[local_block_id0][amp_offset0];
                    CPPCTYPE a1 = decomp_block_pointers[local_block_id0][amp_offset1];
                    CPPCTYPE a2 = decomp_block_pointers[local_block_id1][amp_offset0];
                    CPPCTYPE a3 = decomp_block_pointers[local_block_id1][amp_offset1];
                    CPPCTYPE new_a0 = matrix(0,0) * a0 + matrix(0,1) * a1 + matrix(0,2) * a2 + matrix(0,3) * a3;
                    CPPCTYPE new_a1 = matrix(1,0) * a0 + matrix(1,1) * a1 + matrix(1,2) * a2 + matrix(1,3) * a3;
                    CPPCTYPE new_a2 = matrix(2,0) * a0 + matrix(2,1) * a1 + matrix(2,2) * a2 + matrix(2,3) * a3;
                    CPPCTYPE new_a3 = matrix(3,0) * a0 + matrix(3,1) * a1 + matrix(3,2) * a2 + matrix(3,3) * a3;
                    decomp_block_pointers[local_block_id0][amp_offset0] = new_a0;
                    decomp_block_pointers[local_block_id0][amp_offset1] = new_a1;
                    decomp_block_pointers[local_block_id1][amp_offset0] = new_a2;
                    decomp_block_pointers[local_block_id1][amp_offset1] = new_a3;
                }
            }
        }
    }

#ifdef _USE_MPI
    void set_pair_info(UINT rank, ITYPE step, ITYPE chunk, ITYPE block,
        UINT tgt_outer_rank_gap, UINT offset_mask, UINT swap_width, ITYPE dim_local, 
        UINT& pair_rank, ITYPE& block_id, ITYPE& pair_global_block_id) {
        pair_rank = rank ^ (step << tgt_outer_rank_gap);
        const ITYPE chunk_index = ((rank >> tgt_outer_rank_gap) ^ step) & offset_mask;
        const ITYPE pair_chunk_index = ((pair_rank >> tgt_outer_rank_gap) ^ step) & offset_mask;
        const ITYPE amp_index = (chunk_index ^ (chunk << swap_width)) * dim_local + namps_per_block * block;
        const ITYPE pair_amp_index = (pair_chunk_index ^ (chunk << swap_width)) * dim_local + namps_per_block * block;
        block_id = amp_index >> block_size;
        pair_global_block_id = (pair_amp_index >> block_size) + (nblocks * pair_rank);
    }

    void apply_fused_swap(
        UINT target_qubit_index_0, 
        UINT target_qubit_index_1,
        UINT swap_width
    ) {
        // ex.
        // +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        // |19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
        // +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        // |<-- global qubits-->|<-------     local qubits     ------->|
        // +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        //                    ^^ target=0                ^^target1
        // Fused-SWAP (target0=13, target1=4, block size=4) is same as
        //     SWAP(13,4) + SWAP(14,5) + SWAP(15,6) + SWAP(16,7).
        //
        UINT global_index, local_index;
        if (target_qubit_index_0 > target_qubit_index_1) {
            global_index = target_qubit_index_0;
            local_index = target_qubit_index_1;
        } else {
            global_index = target_qubit_index_1;
            local_index = target_qubit_index_0;
        }
        assert(global_index > (local_index + swap_width - 1));

        /* All remained swaps are pairs of inner and outer */

        MPIutil& mpiutil = MPIutil::get_inst();
        const UINT rank = mpiutil.get_rank();
        const ITYPE dim_local = 1ULL << local_index;
        assert(local_index >= block_size);
        const UINT num_pairs = 1UL << swap_width;
        const UINT tgt_outer_rank_gap = global_index - inner_qc;
        const UINT tgt_inner_rank_gap = inner_qc - local_index;
        UINT pair_rank;
        ITYPE block_id, pair_global_block_id;

        const ITYPE dim_pair = dim >> swap_width;
        const ITYPE num_chunks = dim_pair >> local_index;
        const ITYPE num_loops_per_chunk = dim_local >> block_size;
        UINT offset_mask = (1UL << tgt_inner_rank_gap) - 1;

        
        // // Additional debug output for calculations
        // if (rank == 0) {
        //     std::cout << "Calculated values:" << std::endl;
        //     std::cout << "  local_index: " << local_index << std::endl;
        //     std::cout << "  global_index: " << global_index << std::endl;
        //     std::cout << "  dim_local: " << dim_local << std::endl;
        //     std::cout << "  num_pairs: " << num_pairs << std::endl;
        //     std::cout << "  tgt_outer_rank_gap: " << tgt_outer_rank_gap << std::endl;
        //     std::cout << "  tgt_inner_rank_gap: " << tgt_inner_rank_gap << std::endl;
        //     std::cout << "  dim_pair: " << dim_pair << std::endl;
        //     std::cout << "  num_chunks: " << num_chunks << std::endl;
        //     std::cout << "  num_loops_per_chunk: " << num_loops_per_chunk << std::endl;
        //     std::cout << "  offset_mask: 0x" << std::hex << offset_mask << std::dec << std::endl;
            
        //     // Check for potential issues
        //     if (inner_qc <= local_index) {
        //         std::cout << "  WARNING: inner_qc (" << inner_qc << ") <= local_index (" << local_index << ")" << std::endl;
        //     }
        //     if (num_chunks == 0) {
        //         std::cout << "  ERROR: num_chunks is 0!" << std::endl;
        //     }
        //     if (num_loops_per_chunk == 0) {
        //         std::cout << "  ERROR: num_loops_per_chunk is 0!" << std::endl;
        //     }
        //     if (tgt_inner_rank_gap > 32) {
        //         std::cout << "  WARNING: tgt_inner_rank_gap (" << tgt_inner_rank_gap << ") is too large for shift operation!" << std::endl;
        //     }
        // }

        char* sendbuf;
        size_t sendbytes;
        char* recvbuf[2] = {};
        size_t recvbytes[2] = {};

        const ITYPE sjk_total = (num_pairs - 1) * num_chunks * num_loops_per_chunk;
        ITYPE sjk = 0;
        UINT buf_idx = 0;
        if (0 < sjk_total) {  // first sendrecv
            set_pair_info(rank, 1, 0, 0, tgt_outer_rank_gap, offset_mask, swap_width, dim_local,
                pair_rank, block_id, pair_global_block_id);
            if (block_pointers[block_id] != nullptr) {
                sendbuf = block_pointers[block_id];
                sendbytes = blocks_bytes[block_id];
            } else {
                sendbuf = nullptr;
                sendbytes = 0ULL;
            }
            if (global_blocks_bytes[pair_global_block_id] != 0) {
                recvbuf[buf_idx] = (char*)malloc(global_blocks_bytes[pair_global_block_id]);
                recvbytes[buf_idx] = global_blocks_bytes[pair_global_block_id];
            } else {
                recvbuf[buf_idx] = nullptr;
                recvbytes[buf_idx] = 0UL;
            }
            mpiutil.m_CHAR_isendrecv(sendbuf, sendbytes, recvbuf[buf_idx], recvbytes[buf_idx], pair_rank);
            // std::cout << "rank" << rank << ": Fused-SWAP initial sendrecv local block " << block_id << " and global block " << pair_global_block_id << " (sendbytes " << sendbytes << ", recvbytes " << recvbytes[buf_idx] << ") with rank " << pair_rank << std::endl;
        }
        for (ITYPE step = 1; step < num_pairs; ++step) {  // pair communication
            for (ITYPE j = 0; j < num_chunks; ++j) {
                for (ITYPE k = 0; k < num_loops_per_chunk; ++k) {
                    if ((sjk + 1) < sjk_total) {
                        ITYPE k_next = k + 1;
                        ITYPE j_next = j;
                        ITYPE s_next = step;
                        if (k_next >= num_loops_per_chunk) {
                            k_next = 0;
                            j_next++;
                        }
                        if (j_next >= num_chunks) {
                            j_next = 0;
                            s_next++;
                        }
                        set_pair_info(rank, s_next, j_next, k_next, tgt_outer_rank_gap, offset_mask, 
                            swap_width, dim_local, pair_rank, block_id, pair_global_block_id);
                        if (block_pointers[block_id] != nullptr) {
                            sendbuf = block_pointers[block_id];
                            sendbytes = blocks_bytes[block_id];
                        } else {
                            sendbuf = nullptr;
                            sendbytes = 0UL;
                        }
                        if (global_blocks_bytes[pair_global_block_id] != 0) {
                            recvbuf[buf_idx ^ 1] = (char*)malloc(global_blocks_bytes[pair_global_block_id]);
                            recvbytes[buf_idx ^ 1] = global_blocks_bytes[pair_global_block_id];
                        } else {
                            recvbuf[buf_idx ^ 1] = nullptr;
                            recvbytes[buf_idx ^ 1] = 0UL;
                        }
                        mpiutil.m_CHAR_isendrecv(sendbuf, sendbytes, 
                            recvbuf[buf_idx ^ 1], recvbytes[buf_idx ^ 1], pair_rank);
                        // std::cout << "rank" << rank << ": Fused-SWAP sendrecv local block " << block_id << " and global block " << pair_global_block_id << " (sendbytes " << sendbytes << ", recvbytes " << recvbytes[buf_idx ^ 1] << ") with rank " << pair_rank << std::endl;    
                    }
                    set_pair_info(rank, step, j, k, tgt_outer_rank_gap, offset_mask, swap_width, dim_local,
                        pair_rank, block_id, pair_global_block_id);
                    mpiutil.mpi_wait(2);  // wait async comm
                    // std::cout << "rank" << rank << ": Fused-SWAP completed wait local block " << block_id << " and global block " << pair_global_block_id << " (recvbytes " << recvbytes[buf_idx] << ") from rank " << pair_rank << std::endl;

                    if (recvbuf[buf_idx] == nullptr) {
                        register_SV_block(nullptr, 0UL, block_id);                        
                    } else {
                        register_SV_block(recvbuf[buf_idx], recvbytes[buf_idx], block_id);
                    }
                    buf_idx ^= 1;
                    sjk++;
                }
            }
        }
    }
#endif

    /**
     * \~japanese-en
     * 量子状態をC++の<code>std::complex\<double\></code>の配列として取得する
     *
     * @return 複素ベクトルのポインタ
     */
    virtual CPPCTYPE* data_cpp() const override { return this->_state_vector; }

    /**
     * \~japanese-en 量子状態をcsimのComplex型の配列として取得する
     *
     * @return 複素ベクトルのポインタ
     */
    virtual CTYPE* data_c() const override {
        return reinterpret_cast<CTYPE*>(this->_state_vector);
    }

    /**
     * \~japanese-en 量子状態のノルムを計算する
     *
     * 量子状態のノルムは非ユニタリなゲートを作用した時に小さくなる。
     * @return ノルム
     */
    virtual double get_squared_norm() const override {
        double norm;
#ifdef _USE_MPI
        if (this->outer_qc > 0) {
            norm = state_norm_squared_mpi(this->data_c(), _dim);
        } else
#endif
        {
            norm = state_norm_squared(this->data_c(), _dim);
        }
        return norm;
    }

    /**
     * \~japanese-en 量子状態を正規化する
     *
     * @param norm 自身のノルム
     */
    virtual void normalize(double squared_norm) override {
        ::normalize(squared_norm, this->data_c(), _dim);
    }


    // SPIKE: block-wise sampling over the compressed state.
    // Memory: one decompressed block (2^B amplitudes) at a time.
    // MPI: block masses are all-gathered; every rank draws the same multinomial
    // split of shots over global blocks (same seed), samples its own blocks,
    // and rank 0 gathers the global indices. Other ranks return an empty list.
    std::vector<ITYPE> sampling_blockwise(UINT n_shots, UINT seed) {
        int rank = 0, nranks = 1;
#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        rank = mpiutil.get_rank(); nranks = mpiutil.get_size();
#endif
        auto decompress = [&](UINT b) -> CPPCTYPE* {
#ifdef _USE_GPU_COMP
            return decompress_SV_block_GPU(block_pointers[b], blocks_bytes[b]);
#else
            return decompress_SV_block(block_pointers[b], blocks_bytes[b]);
#endif
        };
        std::vector<double> local_mass(nblocks, 0.0);
        for (UINT b = 0; b < nblocks; b++) {
            if (block_pointers[b] == nullptr) continue;      // all-zero block
            CPPCTYPE* blk = decompress(b);
            double m = 0.0;
            for (UINT j = 0; j < namps_per_block; j++) m += std::norm(blk[j]);
            local_mass[b] = m;
            free(blk);
        }
        std::vector<double> mass(static_cast<size_t>(nblocks) * nranks);
#ifdef _USE_MPI
        MPI_Allgather(local_mass.data(), nblocks, MPI_DOUBLE, mass.data(), nblocks, MPI_DOUBLE, MPI_COMM_WORLD);
#else
        mass = local_mass;
#endif
        double total = 0.0;
        for (double m : mass) total += m;                   // lossy: total is not exactly 1
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::domain_error(
                "CompressedQuantumState::sampling_blockwise: "
                "state has no finite positive probability mass");
        }
        std::mt19937_64 rng(seed);
        std::vector<UINT> shots(mass.size(), 0);
        UINT left = n_shots; double rest = total;
        for (size_t g = 0; g < mass.size() && left > 0; g++) {
            double p = rest > 0 ? std::min(1.0, mass[g] / rest) : 0.0;
            std::binomial_distribution<UINT> bin(left, p);
            UINT k = (g + 1 == mass.size()) ? left : bin(rng);
            shots[g] = k; left -= k; rest -= mass[g];
        }
        std::vector<ITYPE> mine;
        std::mt19937_64 rng_local(seed + 1 + rank);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (UINT b = 0; b < nblocks; b++) {
            UINT k = shots[static_cast<size_t>(rank) * nblocks + b];
            if (k == 0) continue;
            CPPCTYPE* blk = decompress(b);
            std::vector<double> cdf(namps_per_block);
            double acc = 0.0;
            for (UINT j = 0; j < namps_per_block; j++) { acc += std::norm(blk[j]); cdf[j] = acc; }
            free(blk);
            ITYPE base = (static_cast<ITYPE>(rank) << this->inner_qc) + static_cast<ITYPE>(b) * namps_per_block;
            for (UINT s = 0; s < k; s++) {
                double r = uni(rng_local) * acc;
                UINT j = std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
                if (j >= namps_per_block) j = namps_per_block - 1;
                mine.push_back(base + j);
            }
        }
#ifdef _USE_MPI
        int my_n = static_cast<int>(mine.size());
        std::vector<int> counts(nranks), displs(nranks, 0);
        MPI_Gather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<ITYPE> all;
        if (rank == 0) {
            for (int r = 1; r < nranks; r++) displs[r] = displs[r - 1] + counts[r - 1];
            all.resize(displs[nranks - 1] + counts[nranks - 1]);
        }
        static_assert(sizeof(ITYPE) == sizeof(unsigned long long), "ITYPE size");
        MPI_Gatherv(mine.data(), my_n, MPI_UNSIGNED_LONG_LONG, all.data(), counts.data(), displs.data(),
                    MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        if (rank == 0) std::shuffle(all.begin(), all.end(), rng);
        return all;
#else
        std::shuffle(mine.begin(), mine.end(), rng);
        return mine;
#endif
    }

    CPPCTYPE* get_decompressed_state_vector() {
        CPPCTYPE* decomp_block;
        if (this->_state_vector != nullptr) free(this->_state_vector);  // SPIKE: avoid leak on repeated readout
        this->_state_vector = (CPPCTYPE*)malloc(sizeof(CPPCTYPE) * this->_dim);

        for (ITYPE i = 0; i < nblocks; i++) {
#ifdef _USE_GPU_COMP
            decomp_block = decompress_SV_block_GPU(block_pointers[i], blocks_bytes[i]);
#else
            decomp_block = decompress_SV_block(block_pointers[i], blocks_bytes[i]);
#endif
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (ITYPE j = 0; j < namps_per_block; j++) {
                ITYPE index = static_cast<ITYPE>(namps_per_block) * i + j;
                this->_state_vector[index] = decomp_block[j];
            }
            free(decomp_block);
        }

        double norm = get_squared_norm();
        normalize(norm);
        return this->_state_vector;
    }
};

