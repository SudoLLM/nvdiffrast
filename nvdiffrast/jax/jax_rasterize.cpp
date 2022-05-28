#include "../common/rasterize.h"
#include "kernel_helpers.h"
#include "jax_rasterize.h"

RasterizeGLState g_gl_state;

void ThrowIfError(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}


void rasterize_fwd(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    // All inputs and outputs are given in buffers
    // - Inputs
    // TODO: Currently, I only implement instance mode, namely
    // 'pos' in shape (N,V,4), 'tri' in shape (F,3)
    const float   * pos_ptr = reinterpret_cast<const float   *>(buffers[0]);
    const int32_t * tri_ptr = reinterpret_cast<const int32_t *>(buffers[1]);
    const int32_t * ranges_ptr = nullptr;
    // - Outpus
    // TODO: Currently, I only return rast_out, without out_db
    float * out_ptrs[1];
    out_ptrs[0] = reinterpret_cast<float *>(buffers[2]);

    // The size descriptor from opaque
    GemotryDescriptor const & d = *UnpackDescriptor<GemotryDescriptor>(opaque, opaque_len);
    int32_t pos_count = d.N * d.V * 4;
    int32_t tri_count = d.F * 3;
    int32_t vtx_per_inst = d.V;
    int32_t width  = d.W;
    int32_t height = d.H;
    int32_t depth  = d.N;  // instance_mode ? pos.dim_size(0) : ranges.dim_size(0);
    int peeling_idx = -1;  // TODO: input

    // TODO: Gl Context
    // Init context and GL?
    bool init_ctx = !g_gl_state.glFBO;
    if (init_ctx) {
        int cuda_device_idx = 0;  // TODO: get info
        rasterizeInitGLContext(0, g_gl_state, cuda_device_idx); // In common/rasterize.cpp
    }
    else {
        setGLContext(g_gl_state.glctx); // (Re-)Activate GL context.
    }

    // Resize all buffers.
    rasterizeResizeBuffers(0, g_gl_state, pos_count, tri_count, width, height, depth); // In common/rasterize.cpp

    // Newly created GL objects sometimes don't map properly to CUDA until after first context swap. Workaround.
    if (init_ctx) {
        // On first execution, do a bonus context swap.
        releaseGLContext();
        setGLContext(g_gl_state.glctx);
    }

    // Render
    rasterizeRender(
        0, g_gl_state, stream,
        pos_ptr, pos_count, vtx_per_inst,
        tri_ptr, tri_count, ranges_ptr,
        width, height, depth,
        peeling_idx
    );

    // Copy rasterized results into CUDA buffers.
    rasterizeCopyResults(0, g_gl_state, stream, out_ptrs, width, height, depth);

    // Done. Release GL context.
    releaseGLContext();

    ThrowIfError(cudaGetLastError());
}