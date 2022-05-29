// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "jax_rasterize.h"
#include "pybind11_kernel_helpers.h"

namespace py= pybind11;

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["rasterize_fwd"] = EncapsulateFunction(rasterize_fwd);
    return dict;
}

PYBIND11_MODULE(_impl_jax, m) {
    m.def("registrations", &Registrations);
    m.def("build_rasterize_descriptor",
        [](int width, int height, bool enableDB,
           bool instanceMode, int posCount, int triCount, int vtxPerInstance, int depth
        ) {
            return PackDescriptor(RasterizeDescriptor{
                width, height, enableDB,
                instanceMode, posCount, triCount, vtxPerInstance, depth
            });
        },
        py::arg("width"),
        py::arg("height"),
        py::arg("enable_db"),
        py::arg("instance_mode"),
        py::arg("pos_count"),
        py::arg("tri_count"),
        py::arg("vtx_per_instance"),
        py::arg("depth")
    );
}
