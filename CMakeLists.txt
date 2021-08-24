cmake_minimum_required(VERSION 3.9)
project(SYCL_FS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# If you're using the DPCPP compiler, these flags will be used. Set here the devices you want to target
set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Wno-unknown-cuda-version -Xsycl-target-backend=nvptx64-nvidia-cuda-sycldevice --cuda-gpu-arch=sm_75)
#set(DPCPP_FLAGS -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice)
include(cmake/FindSYCL.cmake)

# Default C++ Flags for warnings and optimisation
set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -fno-common -Winit-self -Wuninitialized -Woverloaded-virtual")
set(EXTRA_W_FLAGS "-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5") #-Wnoexcept -Wold-style-cast -Wstrict-null-sentinel -switch-default -Wlogical-op
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-inline-namespace-reopened-noninline -Wno-undef -Wno-unused")
set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")

# Adding the flags to the targets
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS} ${DISABLED_WARNINGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${EXTRA_W_FLAGS} ${DISABLED_WARNINGS} -g -Og -fsanitize=undefined -pthread")

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)

set(LIBRARY_INCLUDES include/async_rpc.hpp include/sycl_fs.hpp)

include(examples/CMakeLists.txt)
