pybind11_add_module(_stator binding.cpp)
target_link_libraries(_stator PRIVATE stator_core stator_physics)

install(TARGETS _stator DESTINATION stator)