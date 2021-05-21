cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME dolfinx-sycl)
project(${PROJECT_NAME})

set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/problem.c src/memory.cpp src/la.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE -fsycl)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)


# Test CSR Matrix
add_executable(test_csr_matrix src/test_csr_matrix.cpp src/problem.c src/memory.cpp src/la.cpp)
target_link_libraries(test_csr_matrix PRIVATE -fsycl )
target_link_libraries(test_csr_matrix PRIVATE dolfinx)


add_subdirectory(src/benchmark)
message(STATUS ${CMAKE_CXX_FLAGS})