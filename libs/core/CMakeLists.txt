add_library(stator_core INTERFACE)      # INTERFACE = no sources, headers only
target_include_directories(stator_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(stator_core INTERFACE linalg)