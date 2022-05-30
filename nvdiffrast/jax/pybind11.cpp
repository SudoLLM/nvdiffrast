// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "jax_rasterize.h"
#include "jax_interpolate.h"
#include "jax_antialias.h"
#include "pybind11_kernel_helpers.h"
#include <glog/logging.h>

namespace py = pybind11;

pybind11::dict Registrations() {
    // TODO: better place to init logging
    // google::InitGoogleLogging("[nvdiffrast]");

    pybind11::dict dict;
    dict["rasterize_fwd"] = EncapsulateFunction(rasterizeFwd);
    dict["rasterize_bwd"] = EncapsulateFunction(rasterizeBwd);
    dict["interpolate_fwd"] = EncapsulateFunction(interpolateFwd);
    dict["interpolate_bwd"] = EncapsulateFunction(interpolateBwd);
    dict["antialias_fwd"] = EncapsulateFunction(antialiasFwd);
    dict["antialias_bwd"] = EncapsulateFunction(antialiasBwd);
    return dict;
}

void RegisterDescriptors(py::module_ & m) {
    m.def("build_rasterize_descriptor",
        [](int numVertices, int numTriangles, int width, int height, bool enableDB,
           bool instanceMode, int posCount, int triCount, int vtxPerInstance, int depth
        ) {
            return PackDescriptor(RasterizeDescriptor{
                numVertices, numTriangles, width, height, enableDB,
                instanceMode, posCount, triCount, vtxPerInstance, depth
            });
        },
        py::arg("num_vertices"),
        py::arg("num_triangles"),
        py::arg("width"),
        py::arg("height"),
        py::arg("enable_db"),
        py::arg("instance_mode"),
        py::arg("pos_count"),
        py::arg("tri_count"),
        py::arg("vtx_per_instance"),
        py::arg("depth")
    );

    m.def("build_interpolate_descriptor", 
        [](int  numVertices, int  numTriangles, int  numAttr,
           int  attrBC, int  width, int  height, int  depth,
           bool enableDB, bool instanceMode,
           bool diffAttrsAll, std::vector<int> & diffAttrsVec
        ) {
            return PackDescriptor(InterpolateDescriptor{
                numVertices, numTriangles, numAttr,
                attrBC, width, height, depth,
                enableDB, instanceMode,
                diffAttrsAll, diffAttrsVec
            });
        },
        py::arg("num_vertices"),
        py::arg("num_triangles"),
        py::arg("num_attrs"),
        py::arg("attr_bc"),
        py::arg("width"),
        py::arg("height"),
        py::arg("depth"),
        py::arg("enable_db"),
        py::arg("instance_mode"),
        py::arg("diff_attrs_all"),
        py::arg("diff_attrs_list")
    );

    m.def("build_antialias_descriptor",
        [](int numVertices, int numTriangles,
           int n, int width, int height, int channels,
           bool instanceMode
        ) {
            return PackDescriptor(AntialiasDescriptor{
                numVertices, numTriangles,
                n, width, height, channels,
                instanceMode
            });
        },
        py::arg("num_vertices"),
        py::arg("num_triangles"),
        py::arg("n"),
        py::arg("width"),
        py::arg("height"),
        py::arg("channels"),
        py::arg("instance_mode")
    );
}

PYBIND11_MODULE(_impl_jax, m) {
    m.def("registrations", &Registrations);
    RegisterDescriptors(m);
}
