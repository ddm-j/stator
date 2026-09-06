cmake_minimum_required(VERSION 3.25)
project(stator CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ENABLE_COVERAGE "Build with gcov instrumentation" OFF)

if(ENABLE_COVERAGE)
  add_compile_options(--coverage -O0 -g)
  add_link_options(--coverage)
endif()

# linalg dependency
include(FetchContent)
FetchContent_Declare(linalg
    GIT_REPOSITORY https://github.com/ddm-j/linalg.git
    GIT_TAG        262a2412063fe067d41adaf2322c669ba9582115
)
FetchContent_MakeAvailable(linalg)

# pybind11
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE
    SYSTEM
)
FetchContent_MakeAvailable(pybind11)

# Libraries
add_subdirectory(libs/core)
add_subdirectory(libs/physics)

# Directories
add_subdirectory(tests)
add_subdirectory(scratch)

# Python Bindings
add_subdirectory(python)