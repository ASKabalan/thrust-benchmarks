# Add CUDA executable


set(MATX_EN_VISUALIZATION ON)
add_subdirectory(pybind11)
add_subdirectory(MatX)


add_executable(matx_gemm ${CMAKE_CURRENT_LIST_DIR}/src/main.cu )
target_link_libraries(matx_gemm matx::matx)
target_include_directories(matx_gemm PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
set_target_properties(matx_gemm PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(matx_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
