#include <cuda.h>
#include "jax_interpolate.h"
#include "kernel_helpers.h"
#include "../common/common.h"
#include "../common/framework.h"
#include "../common/interpolate.h"

//------------------------------------------------------------------------
// Kernel prototypes.

void InterpolateFwdKernel   (const InterpolateKernelParams p);
void InterpolateFwdKernelDa (const InterpolateKernelParams p);
void InterpolateGradKernel  (const InterpolateKernelParams p);
void InterpolateGradKernelDa(const InterpolateKernelParams p);

//------------------------------------------------------------------------
// Helper

static void set_diff_attrs(InterpolateKernelParams& p, bool diff_attrs_all, std::vector<int> const & diff_attrs_vec) {
    if (diff_attrs_all) {
        p.numDiffAttr = p.numAttr;
        p.diff_attrs_all = 1;
    }
    else {
        NVDR_CHECK(diff_attrs_vec.size() <= IP_MAX_DIFF_ATTRS, "too many entries in diff_attrs list (increase IP_MAX_DIFF_ATTRS)");
        p.numDiffAttr = diff_attrs_vec.size();
        memcpy(p.diffAttrs, &diff_attrs_vec[0], diff_attrs_vec.size()*sizeof(int));
    }
}


//------------------------------------------------------------------------
// Forward op.

void interpolateFwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    // Get descriptor
    auto const & d = *UnpackDescriptor<InterpolateDescriptor>(opaque, opaque_len);
    bool enableDA = (d.enableDB) && (d.diffAttrsAll || !d.diffAttrsVec.empty());

    // All inputs and outputs are given in buffers
    // - Inputs
    size_t iBuf = 0;
    const float   * attrPtr   = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * rastPtr   = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * triPtr    = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    const float   * rastDBPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    if (!enableDA) { rastDBPtr = nullptr; }
    // - Outputs
    float * outPtr = reinterpret_cast<float *>(buffers[iBuf++]); // out
    float * outDAPtr = reinterpret_cast<float *>(buffers[iBuf++]); // out_da
    if (!enableDA) { outDAPtr = nullptr; }

    // TODO: check types and shapes in python code

    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    p.instance_mode = d.instanceMode;

    // Extract input dimensions.
    p.numVertices  = d.numVertices;
    p.numTriangles = d.numTriangles;
    p.numAttr      = d.numAttr;
    p.height       = d.height;
    p.width        = d.width;
    p.depth        = d.depth;

    // Set attribute pixel differential info if enabled, otherwise leave as zero.
    if (enableDA) { set_diff_attrs(p, d.diffAttrsAll, d.diffAttrsVec); }
    else          { p.numDiffAttr = 0; }

    // Get input pointers.
    p.attr   = attrPtr;
    p.rast   = rastPtr;
    p.tri    = triPtr;
    p.rastDB = rastDBPtr;
    p.attrBC = d.attrBC;

    // Get output tensors.
    p.out = outPtr;
    p.outDA = outDAPtr;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.rast   & 15), "rast input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rastDB & 15), "rast_db input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.outDA  &  7), "out_da output tensor not aligned to float2");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    void* func = enableDA ? (void*)InterpolateFwdKernelDa : (void*)InterpolateFwdKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}


void interpolateBwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    // Get descriptor
    auto const & d = *UnpackDescriptor<InterpolateDescriptor>(opaque, opaque_len);
    bool enableDA = (d.enableDB) && (d.diffAttrsAll || !d.diffAttrsVec.empty());

    // All inputs and outputs are given in buffers
    // - Inputs
    size_t iBuf = 0;
    const float   * attrPtr   = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * rastPtr   = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * triPtr    = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    const float   * dyPtr     = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * rastDBPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * ddaPtr    = reinterpret_cast<const float   *>(buffers[iBuf++]);
    if (!enableDA) { rastDBPtr = nullptr; ddaPtr = nullptr; }
    // - Outputs
    float * gradAttrPtr = reinterpret_cast<float *>(buffers[iBuf++]);
    float * gradRastPtr = reinterpret_cast<float *>(buffers[iBuf++]);
    float * gradRastDBPtr = reinterpret_cast<float *>(buffers[iBuf++]);
    if (!enableDA) { gradRastDBPtr = nullptr; }

    // TODO: check types and shapes in python code

    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    p.instance_mode = d.instanceMode;

    // Extract input dimensions.
    p.numVertices  = d.numVertices;
    p.numTriangles = d.numTriangles;
    p.numAttr      = d.numAttr;
    p.height       = d.height;
    p.width        = d.width;
    p.depth        = d.depth;
    p.attrBC       = d.attrBC;
    // int attrDepth = d.instanceMode ? (attr.sizes().size() > 1 ? attr.size(0) : 0) : 1;
    // p.attrBC = (p.instance_mode && attr_depth < p.depth) ? 1 : 0;

    // Set attribute pixel differential info if enabled, otherwise leave as zero.
    if (enableDA) { set_diff_attrs(p, d.diffAttrsAll, d.diffAttrsVec); }
    else          { p.numDiffAttr = 0; }

    // Get input pointers.
    p.attr = attrPtr;
    p.rast = rastPtr;
    p.tri = triPtr;
    p.dy = dyPtr;
    p.rastDB = rastDBPtr;
    p.dda = ddaPtr;

    // Get output pointers
    p.gradAttr = gradAttrPtr;
    p.gradRaster = gradRastPtr;
    p.gradRasterDB = gradRastDBPtr;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.rast         & 15), "rast input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rastDB       & 15), "rast_db input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.dda          &  7), "dda input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p.gradRaster   & 15), "grad_rast output tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.gradRasterDB & 15), "grad_rast_db output tensor not aligned to float4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(IP_GRAD_MAX_KERNEL_BLOCK_WIDTH, IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    void* func = enableDA ? (void*)InterpolateGradKernelDa : (void*)InterpolateGradKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}
