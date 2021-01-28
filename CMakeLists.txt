cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME dolfinx-sycl)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG ${CMAKE_CXX_FLAGS} -Wall")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(DOLFINX REQUIRED)
find_package(Ginkgo REQUIRED)

set(supported_implementations
  ComputeCpp
  hipSYCL
  LLVM
  LLVM-CUDA
)

list(FIND supported_implementations ${SYCL_IMPL} impl_idx)
if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
  message(FATAL_ERROR "Please specify SYCL_IMPL (one of: ${supported_implementations})")
endif()

if(SYCL_IMPL STREQUAL "ComputeCpp")
  find_package(ComputeCpp REQUIRED)
elseif(SYCL_IMPL STREQUAL "hipSYCL")
  find_package(hipSYCL CONFIG REQUIRED)
  if($ENV{HIPSYCL_PLATFORM} STREQUAL cuda)
    message(STATUS "Compiling for cuda devices!")
  endif()
elseif(SYCL_IMPL STREQUAL "LLVM")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
elseif(SYCL_IMPL STREQUAL "LLVM-CUDA")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice --cuda-path=${CUDA_PATH}")
  message(STATUS "Compiling for cuda devices!")
endif()

add_executable(${PROJECT_NAME} 
  src/main.cpp 
  src/problem.c 
  src/assemble_impl.cpp 
  src/la.cpp 
  src/memory.cpp
  src/solve.cpp
  )

add_executable(${PROJECT_NAME}-lookup
  src/main.cpp 
  src/problem.c 
  src/assemble_impl.cpp 
  src/la.cpp 
  src/memory.cpp
  src/solve.cpp
  )

add_executable(${PROJECT_NAME}-search
  src/main.cpp 
  src/problem.c 
  src/assemble_impl.cpp 
  src/la.cpp 
  src/memory.cpp 
  src/solve.cpp
  )

target_link_libraries(${PROJECT_NAME} PRIVATE Ginkgo::ginkgo dolfinx)
target_link_libraries(${PROJECT_NAME}-lookup PRIVATE Ginkgo::ginkgo dolfinx)
set_target_properties(${PROJECT_NAME}-lookup PROPERTIES COMPILE_FLAGS "-DUSE_ATOMICS_LOOKUP")
target_link_libraries(${PROJECT_NAME}-search PRIVATE Ginkgo::ginkgo dolfinx)
set_target_properties(${PROJECT_NAME}-search PROPERTIES COMPILE_FLAGS "-DUSE_ATOMICS_SEARCH")


if(SYCL_IMPL STREQUAL "LLVM")
  target_compile_definitions(${PROJECT_NAME} PRIVATE __LLVM_SYCL__)
  target_compile_definitions(${PROJECT_NAME}-lookup PRIVATE __LLVM_SYCL__)
  target_compile_definitions(${PROJECT_NAME}-search PRIVATE __LLVM_SYCL__)
elseif(SYCL_IMPL STREQUAL "LLVM-CUDA")
  target_compile_definitions(${PROJECT_NAME} PRIVATE __LLVM_SYCL__)
  target_compile_definitions(${PROJECT_NAME}-lookup PRIVATE __LLVM_SYCL__)
  target_compile_definitions(${PROJECT_NAME}-search PRIVATE __LLVM_SYCL__)
elseif(SYCL_IMPL STREQUAL "hipSYCL")
  add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES src/main.cpp src/main_atomic.cpp  src/assemble_impl.cpp src/la.cpp src/memory.cpp)
  add_sycl_to_target(TARGET ${PROJECT_NAME}-lookup SOURCES src/main.cpp src/main_atomic.cpp  src/assemble_impl.cpp src/la.cpp src/memory.cpp)
  add_sycl_to_target(TARGET ${PROJECT_NAME}-search SOURCES src/main.cpp src/main_atomic.cpp  src/assemble_impl.cpp src/la.cpp src/memory.cpp)
endif()


add_subdirectory(src/benchmark)

message(STATUS ${CMAKE_CXX_FLAGS})