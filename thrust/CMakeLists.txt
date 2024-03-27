find_package(Thrust)

thrust_create_target(ThrustCuda FROM_OPTIONS
                     HOST CPP
                     DEVICE CUDA
)

add_executable(thrust_gemm ${CMAKE_CURRENT_LIST_DIR}/src/thrust_mm.cu)
target_link_libraries(thrust_gemm PRIVATE NVHPC::CUDA ThrustCuda)
set_target_properties(thrust_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})