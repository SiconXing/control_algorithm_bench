cmake_minimum_required(VERSION 3.14)
project(function_env)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Discover pybind11 installed via pip
execute_process(
    COMMAND python3 -c
    "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# Build the Python extension module
pybind11_add_module(function_env
    src/function_env.cpp
    src/bindings.cpp
)

# Set output directory so the .so file lands in the project root
set_target_properties(function_env PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
