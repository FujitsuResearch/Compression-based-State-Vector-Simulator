cmake_minimum_required(VERSION 3.5)

file(GLOB VQCSIM_SRC
    "*.cpp"
    "*.c"
    "*.hpp"
    "*.h")

add_library(vqcsim_static STATIC ${VQCSIM_SRC})
add_library(vqcsim_shared SHARED ${VQCSIM_SRC})
target_link_libraries(vqcsim_static cppsim_static)
target_link_libraries(vqcsim_shared cppsim_static)

if(USE_MPI)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(vqcsim_static MPI::MPI_CXX)
        target_link_libraries(vqcsim_shared MPI::MPI_CXX)
    else()
        target_link_libraries(vqcsim_static ${MPI_LIBRARIES})
        target_link_libraries(vqcsim_shared ${MPI_LIBRARIES})
    endif()
endif()

set_target_properties(vqcsim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
