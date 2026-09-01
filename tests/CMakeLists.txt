# --- environment: gtest (once) ---
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  SYSTEM)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# --- shared test compile flags ---
add_library(stator_test_flags INTERFACE)
target_compile_options(stator_test_flags INTERFACE
  -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror)

# --- the pattern, as a function ---
function(stator_add_test name module_lib)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(${name} PRIVATE
    GTest::gmock_main stator_test_flags ${module_lib})
  gtest_discover_tests(${name})
  set_property(GLOBAL APPEND PROPERTY STATOR_TEST_TARGETS ${name})
endfunction()

# --- modules ---
add_subdirectory(core)

# --- coverage: last, after all test targets exist ---
if(ENABLE_COVERAGE)
  find_program(LCOV lcov REQUIRED)
  find_program(GENHTML genhtml REQUIRED)
  get_property(all_tests GLOBAL PROPERTY STATOR_TEST_TARGETS)
  add_custom_target(coverage
    COMMAND ${LCOV} --directory . --zerocounters
    COMMAND ctest --output-on-failure
    COMMAND ${LCOV} --directory . --capture
            --output-file coverage.info
            --rc branch_coverage=1 --ignore-errors mismatch
    COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/_deps/*'
            --output-file coverage_filtered.info
            --rc branch_coverage=1 --ignore-errors mismatch
    COMMAND ${GENHTML} --demangle-cpp --branch-coverage
            -o coverage coverage_filtered.info
    COMMENT "Running tests and generating coverage report"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${all_tests})
endif()