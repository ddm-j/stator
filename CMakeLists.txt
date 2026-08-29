cmake_minimum_required(VERSION 3.25)
project(stator CXX)

# Options
option(STATOR_BUILD_TEST "Build/run the tests" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(libs/core)

# Directories
if (STATOR_BUILD_TEST)
    add_subdirectory(tests)
endif()
add_subdirectory(scratch)