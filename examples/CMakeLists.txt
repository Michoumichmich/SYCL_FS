
include_directories(examples/includes)

add_executable(bmp_processing_example examples/bmp_processing.cpp ${LIBRARY_INCLUDES})
add_sycl_to_target(TARGET bmp_processing_example SOURCES examples/bmp_processing.cpp)

add_executable(bmp_processing_example_better examples/bmp_processing_better.cpp ${LIBRARY_INCLUDES})
add_sycl_to_target(TARGET bmp_processing_example_better SOURCES examples/bmp_processing_better.cpp)

add_executable(demo_fs examples/demo_fs.cpp ${LIBRARY_INCLUDES})
add_sycl_to_target(TARGET demo_fs SOURCES examples/demo_fs.cpp)

add_executable(demo_rpc examples/demo_rpc.cpp ${LIBRARY_INCLUDES})
add_sycl_to_target(TARGET demo_rpc SOURCES examples/demo_rpc.cpp)

add_executable(fs_benchmark examples/fs_benchmark.cpp ${LIBRARY_INCLUDES})
add_sycl_to_target(TARGET fs_benchmark SOURCES examples/fs_benchmark.cpp)

add_executable(random_walk examples/random_walk.cpp ${LIBRARY_INCLUDES})
add_sycl_to_target(TARGET random_walk SOURCES examples/random_walk.cpp)