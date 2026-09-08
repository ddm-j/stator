# Header self-test.
#
# Compiles every public header as the SOLE include of its own translation unit.
# A header that only works because some other header happened to be included
# first will fail here, at the commit that broke it, instead of months later
# when someone reorders two include lines.
#
# The .cpp files are generated at configure time into the build tree; nothing
# is checked in. CONFIGURE_DEPENDS re-runs the glob on every build, so a new
# header picks up a self-test automatically.

option(ENABLE_HEADER_SELFTEST "Compile each public header standalone" ON)

if(NOT ENABLE_HEADER_SELFTEST)
    return()
endif()

set(selftest_roots
    ${CMAKE_SOURCE_DIR}/libs/core/include
    ${CMAKE_SOURCE_DIR}/libs/physics/include
)

set(selftest_srcs "")
foreach(root IN LISTS selftest_roots)
    file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${root}/*.h)
    foreach(header IN LISTS headers)
        file(RELATIVE_PATH inc ${root} ${header})
        string(MAKE_C_IDENTIFIER ${inc} stem)
        file(CONFIGURE OUTPUT generated/${stem}.cpp CONTENT "#include <${inc}>\n")
        list(APPEND selftest_srcs ${CMAKE_CURRENT_BINARY_DIR}/generated/${stem}.cpp)
    endforeach()
endforeach()

# OBJECT library: compiled, never linked, so no main() is needed.
add_library(header_selftest OBJECT ${selftest_srcs})
target_link_libraries(header_selftest PRIVATE stator_core stator_physics)
