# Executable
add_executable(benchmark main.cpp ${CMAKE_SOURCE_DIR}/src/problem.c)

# Set C++17 standard
target_compile_features(benchmark PRIVATE cxx_std_17)

# Target libraries
target_link_libraries(benchmark dolfinx)
set_target_properties(benchmark PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}")