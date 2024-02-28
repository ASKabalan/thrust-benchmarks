# Add CUDA executable
add_executable(cuda_bencher ${CMAKE_CURRENT_LIST_DIR}/src/cuda_bencher.cu)
target_link_libraries(cuda_bencher PRIVATE NVHPC::CUDA)
set_target_properties(cuda_bencher PROPERTIES LINKER_LANGUAGE CXX)