cmake_minimum_required(VERSION 3.12...3.18)
project(_impl_jax LANGUAGES CXX)

enable_language(CUDA)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(pybind11_DIR ~/work/.local/lib/python3.7/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/../common/common.cpp
    ${PROJECT_SOURCE_DIR}/../common/glutil.cpp
    ${PROJECT_SOURCE_DIR}/../common/rasterize.cu
    ${PROJECT_SOURCE_DIR}/../common/rasterize.cpp
    ${PROJECT_SOURCE_DIR}/pybind11.cpp
    ${PROJECT_SOURCE_DIR}/jax_rasterize.cpp
    ${PROJECT_SOURCE_DIR}/jax_interpolate.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PUBLIC GL EGL glog cuda)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DNVDR_JAX")

# install(TARGETS ${PROJECT_NAME} DESTINATION kepler_jax)