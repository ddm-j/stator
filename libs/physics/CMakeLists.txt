add_library(stator_physics INTERFACE)      # INTERFACE = no sources, headers only
target_include_directories(stator_physics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(stator_physics INTERFACE stator::core)