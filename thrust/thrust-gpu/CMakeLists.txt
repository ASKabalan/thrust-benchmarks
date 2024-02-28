find_package(Thrust)

thrust_create_target(ThrustCuda FROM_OPTIONS
                     HOST CPP
                     DEVICE CUDA
)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../src/thrust_bencher.cu.cpp PROPERTIES LANGUAGE CUDA)
add_executable(thrust_gpu ${CMAKE_CURRENT_LIST_DIR}/../src/thrust_bencher.cu.cpp)
target_link_libraries(thrust_gpu PRIVATE NVHPC::CUDA ThrustCuda)