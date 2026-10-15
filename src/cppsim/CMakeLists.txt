cmake_minimum_required(VERSION 3.5)

if (USE_GPU_COMP)
    file(GLOB CPPSIM_SRC
        "*.cpp"
        "*.hpp"
        "*.cu"
        "*.cuh"
    )
else()
    file(GLOB CPPSIM_SRC
        "*.cpp"
        "*.hpp"
    )
endif()
    
if (USE_GPU_COMP)
    find_package(CUDA REQUIRED)
    find_package(nvcomp CONFIG REQUIRED)
    if (USE_RMM)
        find_package(rmm REQUIRED)
    endif()

    # Auto-detect CUDA compute capability using nvidia-smi
    execute_process(
        COMMAND nvidia-smi --id 0 --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(GPU_COMPUTE_CAP)
        # Convert decimal format (e.g., "9.0") to architecture format (e.g., "90")
        string(REPLACE "." "" CUDA_ARCH ${GPU_COMPUTE_CAP})
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
        message(STATUS "Auto-detected CUDA architecture: ${CUDA_ARCH}")
    else()
        message(WARNING "Could not auto-detect CUDA architecture, using default: 90")
        set(CMAKE_CUDA_ARCHITECTURES 90)
    endif()

    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    enable_language(CUDA)

    add_library(cppsim_static STATIC ${CPPSIM_SRC})
    add_library(cppsim_shared SHARED ${CPPSIM_SRC})
    target_link_libraries(cppsim_static csim_static CUDA::cudart CUDA::cublas CUDA::curand nvcomp::nvcomp)
    target_link_libraries(cppsim_shared csim_static CUDA::cudart CUDA::cublas CUDA::curand nvcomp::nvcomp)
    if(USE_RMM)
        target_link_libraries(cppsim_static csim_static rmm::rmm)
        target_link_libraries(cppsim_shared csim_static rmm::rmm)
    endif()
    if(USE_MPI)
        if(TARGET MPI::MPI_CXX)
            target_link_libraries(cppsim_static MPI::MPI_CXX)
            target_link_libraries(cppsim_shared MPI::MPI_CXX)
        else()
            target_link_libraries(cppsim_static ${MPI_LIBRARIES})
            target_link_libraries(cppsim_shared ${MPI_LIBRARIES})
        endif()
    endif()
    target_include_directories(cppsim_static PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_include_directories(cppsim_shared PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(cppsim_static PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
    target_compile_options(cppsim_shared PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
else()
    add_library(cppsim_static STATIC ${CPPSIM_SRC})
    add_library(cppsim_shared SHARED ${CPPSIM_SRC})
    target_link_libraries(cppsim_static csim_static)
    target_link_libraries(cppsim_shared csim_static)
endif()

if(USE_MPI)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(cppsim_static MPI::MPI_CXX)
        target_link_libraries(cppsim_shared MPI::MPI_CXX)
    else()
        target_link_libraries(cppsim_static ${MPI_LIBRARIES})
        target_link_libraries(cppsim_shared ${MPI_LIBRARIES})
    endif()
endif()

add_dependencies(cppsim_static eigen)
add_dependencies(cppsim_shared eigen)

if (USE_GPU)
    target_link_libraries(cppsim_static gpusim_static)
    target_link_libraries(cppsim_shared gpusim_static)
endif()

set_target_properties(cppsim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)