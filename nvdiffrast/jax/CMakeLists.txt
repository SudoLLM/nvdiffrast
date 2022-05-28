cmake_minimum_required(VERSION 3.12...3.18)
project(_impl_jax LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(pybind11_DIR ~/work/.local/lib/python3.7/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)

# find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/pybind11.cpp
    ${PROJECT_SOURCE_DIR}/../common/common.cpp
    ${PROJECT_SOURCE_DIR}/../common/glutil.cpp
    ${PROJECT_SOURCE_DIR}/../common/rasterize.cu
    ${PROJECT_SOURCE_DIR}/../common/rasterize.cpp
    ${PROJECT_SOURCE_DIR}/jax_rasterize.cu
)
target_link_libraries(${PROJECT_NAME} PRIVATE GL EGL)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DNVDR_JAX -DGL_VERSION_3_2")
# install(TARGETS ${PROJECT_NAME} DESTINATION kepler_jax)