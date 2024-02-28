find_package(Thrust)

thrust_create_target(ThrustMP FROM_OPTIONS
                     HOST CPP
                     DEVICE OMP
)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../src/thrust_bencher.cu.cpp PROPERTIES LANGUAGE CXX)
add_executable(thrust_cpu ${CMAKE_CURRENT_LIST_DIR}/../src/thrust_bencher.cu.cpp)
target_link_libraries(thrust_cpu PRIVATE MPI::MPI_CXX ThrustMP)