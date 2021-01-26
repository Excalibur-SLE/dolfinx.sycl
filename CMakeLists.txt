cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME dolfinx-sycl)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG ${CMAKE_CXX_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)
find_package(Ginkgo REQUIRED)
find_package(PETSc 3.13)
find_package(MPI REQUIRED)
find_package(Eigen3 3.3.9 REQUIRED)

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

if(SYCL_IMPL STREQUAL "hipSYCL")
  find_package(hipSYCL CONFIG REQUIRED)
  find_package(OpenMP)
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

set(LIB_SOURCES
  ${LIB_SOURCES}
  src/assemble_impl.cpp 
  src/la.cpp 
  src/memory.cpp 
  src/timing.cpp 
  src/solve.cpp
  src/problem.c
  )

set(LIB_HEADERS 
  ${LIB_HEADERS}
  src/assemble_impl.hpp
  src/la.hpp 
  src/memory.hpp 
  src/timing.hpp
  src/solve.hpp
  src/problem.h
  )

add_library(${PROJECT_NAME} ${LIB_SOURCES} ${LIB_HEADERS})
target_link_libraries(${PROJECT_NAME} PUBLIC  dolfinx)
target_link_libraries(${PROJECT_NAME} PUBLIC Ginkgo::ginkgo)
target_link_libraries(${PROJECT_NAME} PUBLIC PETSC::petsc)
target_link_libraries(${PROJECT_NAME} PRIVATE PETSC::petsc_static)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

set(EXECUTABLES
  two-stage
  lookup-table
  binary-search
)

add_executable(two-stage src/main.cpp)
target_link_libraries(two-stage SHARED ${PROJECT_NAME})

add_executable(lookup-table src/main.cpp)
target_link_libraries(lookup-table PUBLIC ${PROJECT_NAME})
set_target_properties(lookup-table PROPERTIES COMPILE_FLAGS "-DUSE_ATOMICS_LOOKUP")

add_executable(binary-search src/main.cpp)
target_link_libraries(binary-search PUBLIC ${PROJECT_NAME})
set_target_properties(binary-search PROPERTIES COMPILE_FLAGS "-DUSE_ATOMICS_SEARCH")

if(SYCL_IMPL STREQUAL "LLVM")
  foreach(EXECUTABLE ${EXECUTABLES})
    target_compile_definitions(${EXECUTABLE} PRIVATE __LLVM_SYCL__)
  endforeach(EXECUTABLE ${EXECUTABLES})
elseif(SYCL_IMPL STREQUAL "hipSYCL")
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
  add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${LIB_SOURCES})
  foreach(EXECUTABLE ${EXECUTABLES})
    add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES src/main.cpp)
  endforeach(EXECUTABLE ${EXECUTABLES})
endif()

add_subdirectory(src/benchmark)

message(STATUS ${CMAKE_CXX_FLAGS})