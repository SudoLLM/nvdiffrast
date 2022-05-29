#include "../common/rasterize.h"
#include "kernel_helpers.h"
#include "jax_rasterize.h"

RasterizeGLState g_glState;  // TODO: per-device gl state

void rasterize_fwd(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    // Get descriptor
    RasterizeDescriptor const & d = *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);
    int peelingIdx = -1;  // TODO: input
    int cudaDeviceIdx = 0;  // TODO: get cuda device index

    // Init context and GL?
    g_glState.enableDB = d.enableDB;
    bool initCtx = !g_glState.glFBO;
    if (initCtx) {
        rasterizeInitGLContext(NVDR_CTX_PARAMS, g_glState, cudaDeviceIdx);
    } else {
        setGLContext(g_glState.glctx); // (Re-)Activate GL context.
    }

    // Resize all buffers.
    rasterizeResizeBuffers(NVDR_CTX_PARAMS, g_glState, d.posCount, d.triCount, d.width, d.height, d.depth);

    // Newly created GL objects sometimes don't map properly to CUDA until after first context swap. Workaround.
    if (initCtx) {
        // On first execution, do a bonus context swap.
        releaseGLContext();
        setGLContext(g_glState.glctx);
    }

    // All inputs and outputs are given in buffers
    // - Inputs
    size_t iBuf = 0;
    const float   * posPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * triPtr = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    const int32_t * rangesPtr = (d.instanceMode)
        ? nullptr
        : reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    // - Outputs
    float * outPtrs[2];
    outPtrs[0] = reinterpret_cast<float *>(buffers[iBuf++]);
    outPtrs[1] = reinterpret_cast<float *>(buffers[iBuf++]);
    if (!d.enableDB) outPtrs[1] = nullptr;

    // Render
    rasterizeRender(
        NVDR_CTX_PARAMS, g_glState, stream,
        posPtr, d.posCount, d.vtxPerInstance,
        triPtr, d.triCount, rangesPtr,
        d.width, d.height, d.depth,
        peelingIdx
    );

    // Copy rasterized results into CUDA buffers.
    rasterizeCopyResults(0, g_glState, stream, outPtrs, d.width, d.height, d.depth);

    // Done. Release GL context.
    releaseGLContext();
}