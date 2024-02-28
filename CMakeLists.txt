# Minimum required CMake version
cmake_minimum_required(VERSION 3.20...3.22)

set(CMAKE_CXX_STANDARD 17)
# Project name
project(thrust_benchmarking LANGUAGES CXX CUDA)

set(NVHPC_CUDA_VERSION 11.8)
# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)
# Add Thrust and NVHPC cmake to cmake prefix path from $NVHPC_ROOT/cmake and thrust from $CUDA_HOME/lib64/cmake/thrust
list(APPEND CMAKE_PREFIX_PATH $ENV{NVHPC_ROOT}/cmake)
list(APPEND CMAKE_PREFIX_PATH $ENV{CUDA_HOME}/lib64/cmake/thrust)

find_package(NVHPC REQUIRED COMPONENTS MATH CUDA MPI)


string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib" "/include" NVHPC_MPI_INCLUDE_DIR ${NVHPC_MPI_LIBRARY_DIR})
string(REPLACE "lib64" "bin/nvcc" NVHPC_CUDA_COMPILER ${NVHPC_CUDA_LIBRARY_DIR})

set(CMAKE_CUDA_COMPILER ${NVHPC_CUDA_COMPILER})

# Set includes for all targets

include_directories( ${CMAKE_CURRENT_LIST_DIR}/include 
                     ${NVHPC_CUDA_INCLUDE_DIR}
                     ${NVHPC_MATH_INCLUDE_DIR}
                     ${Thrust_INCLUDE_DIRS}
                     ${NVHPC_MPI_INCLUDE_DIR}            
                     )
        

add_subdirectory(standard)                    
add_subdirectory(thrust)
add_subdirectory(cuda)                    

# Add Thrust GPU executable
#set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/thrust_bencher.cu.cpp PROPERTIES LANGUAGE CUDA)
#add_executable(thrust_gpu SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/thrust_bencher.cu.cpp CMAKE_ARGS -DCMAKE_CUDA_COMPILER=nvcc)
#target_link_libraries(thrust_gpu PRIVATE NVHPC::CUDA ThrustCuda)

# Add Thrust OMP CPU executable
#set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/thrust_bencher.cu.cpp PROPERTIES LANGUAGE CXX)
#add_executable(thrust_cpu ${CMAKE_CURRENT_LIST_DIR}/src/thrust_bencher.cu.cpp)
#target_link_libraries(thrust_cpu PRIVATE ThrustMP MPI::MPI_CXX)

# Add CUDA executable
