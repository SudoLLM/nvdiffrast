#include <cuda.h>
#include "kernel_helpers.h"
#include "jax_rasterize.h"
#include "../common/common.h"
#include "../common/rasterize.h"

RasterizeGLState g_glState;  // TODO: per-device gl state

static int getCudaDeviceId() {
    CUdevice cuDev;
    auto cuRes = cuCtxGetDevice(&cuDev);
    CHECK_EQ(cuRes, CUDA_SUCCESS) << "Cuda error: " << cudaGetLastError() << "[cuCtxGetDevice(CUdevice*);]";
    return (int)cuDev;
}

void rasterizeFwd(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) { 
    // Get descriptor
    RasterizeDescriptor const & d = *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);
    int peelingIdx = -1;  // TODO: input

    // Init context and GL?
    g_glState.enableDB = d.enableDB;
    bool initCtx = !g_glState.glFBO;
    if (initCtx) {
        int cudaDeviceIdx = getCudaDeviceId();
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
    const int32_t * rangesPtr = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    if (d.instanceMode) { rangesPtr = nullptr; }
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


// Kernel prototypes.
void RasterizeGradKernel(const RasterizeGradParams p);
void RasterizeGradKernelDb(const RasterizeGradParams p);

void rasterizeBwd(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    // Get descriptor
    RasterizeDescriptor const & d = *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);

    // Inputs
    size_t iBuf = 0;
    const float   * posPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * triPtr = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    const float   * outPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * dyPtr  = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * ddbPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    // Outputs
    float * gradPtr = reinterpret_cast<float *>(buffers[iBuf++]);

    // TODO: check device

    RasterizeGradParams p = {};
    memset(&p, 0, sizeof(p));  // TODO: necessary?

    // Determine instance mode.
    p.instance_mode = (d.instanceMode) ? 1 : 0;

    // Shape is taken from the descriptor, checked in python code
    p.depth  = d.depth;
    p.height = d.height;
    p.width  = d.width;

    // Populate parameters.
    p.numTriangles = d.numTriangles;
    p.numVertices = d.numVertices;

    // Set up pixel position to clip space x, y transform.
    p.xs = 2.f / (float)p.width;
    p.xo = 1.f / (float)p.width - 1.f;
    p.ys = 2.f / (float)p.height;
    p.yo = 1.f / (float)p.height - 1.f;

    // Input data pointers
    p.pos = posPtr;
    p.tri = triPtr;
    p.out = outPtr;
    p.dy  = dyPtr;
    p.ddb = (d.enableDB) ? ddbPtr : NULL;

    // Output data pointers
    p.grad = gradPtr;

    // Clear the output buffers.
    size_t gradBytes = (p.instance_mode ? p.depth : 1) * p.numVertices * 4 * sizeof(float);
    cudaMemsetAsync(p.grad, 0, gradBytes, stream);

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos & 15), "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.dy  &  7), "dy input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p.ddb & 15), "ddb input tensor not aligned to float4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH, RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    void* func = (d.enableDB) ? (void*)RasterizeGradKernelDb : (void*)RasterizeGradKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}