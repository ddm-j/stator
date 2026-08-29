add_executable(scratch main.cpp)
target_link_libraries(scratch PRIVATE stator_core)

target_compile_options(scratch
    PRIVATE
    -Wall
    -Weffc++
    -Wextra
    -Wconversion
    -Wsign-conversion
    -Werror
)