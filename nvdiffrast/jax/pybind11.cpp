// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "jax_rasterize.h"
#include "pybind11_kernel_helpers.h"

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["rasterize_fwd"] = EncapsulateFunction(rasterize_fwd);
    return dict;
}

PYBIND11_MODULE(_impl_jax, m) {
    m.def("registrations", &Registrations);
    m.def("build_descriptor", [](int32_t W, int32_t H, int32_t N, int32_t V, int32_t F) {
        return PackDescriptor(GemotryDescriptor{W, H, N, V, F});
    });
}
