cmake_minimum_required(VERSION 3.12)
project(_impl_jax LANGUAGES CXX)

# sudo apt update
# sudo apt install libegl1-mesa-dev libgoogle-glog-dev -y

enable_language(CUDA)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/../common/common.cpp
    ${PROJECT_SOURCE_DIR}/../common/glutil.cpp
    ${PROJECT_SOURCE_DIR}/../common/rasterize.cu
    # ${PROJECT_SOURCE_DIR}/../common/rasterize.cpp
    ${PROJECT_SOURCE_DIR}/../common/cudaraster/impl/Buffer.cpp
    ${PROJECT_SOURCE_DIR}/../common/cudaraster/impl/CudaRaster.cpp
    ${PROJECT_SOURCE_DIR}/../common/cudaraster/impl/RasterImpl.cu
    ${PROJECT_SOURCE_DIR}/../common/cudaraster/impl/RasterImpl.cpp
    ${PROJECT_SOURCE_DIR}/../common/interpolate.cu
    ${PROJECT_SOURCE_DIR}/../common/antialias.cu
    ${PROJECT_SOURCE_DIR}/../common/texture.cu
    ${PROJECT_SOURCE_DIR}/../common/texture.cpp
    ${PROJECT_SOURCE_DIR}/pybind11.cpp
    ${PROJECT_SOURCE_DIR}/jax_rasterize_cuda.cpp
    # ${PROJECT_SOURCE_DIR}/jax_rasterize_gl.cpp
    ${PROJECT_SOURCE_DIR}/jax_interpolate.cpp
    ${PROJECT_SOURCE_DIR}/jax_antialias.cpp
    ${PROJECT_SOURCE_DIR}/jax_texture.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries     (${PROJECT_NAME} PUBLIC GL EGL glog cuda)
target_compile_features   (${PROJECT_NAME} PUBLIC cxx_std_11)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DNVDR_JAX")
