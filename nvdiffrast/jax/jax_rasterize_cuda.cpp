#include <cuda.h>
#include <memory>
#include "kernel_helpers.h"
#include "jax_rasterize.h"
#include "../common/common.h"
#include "../common/rasterize.h"
#include "../common/cudaraster/CudaRaster.hpp"
#include "../common/cudaraster/impl/Constants.hpp"

// Kernel prototypes.
void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p);
void RasterizeGradKernel(const RasterizeGradParams p);
void RasterizeGradKernelDb(const RasterizeGradParams p);

// TODO: support different instance?
std::unqiue_ptr<CR::CudaRaster> g_cr(nullptr);

void rasterizeFwd(
    cudaStream_t stream,
    void      ** buffers,
    const char * opaque,
    std::size_t  opaque_len
) { 
    // Get descriptor
    auto const & d = *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);
    // TODO: support different instance?
    if (!g_cr) {
        g_cr = std::make_unique<CR::CudaRaster>();
    }
    auto * cr = g_cr.get();

    // All inputs and outputs are given in buffers.
    size_t iBuf = 0;
    // - Inputs
    const float   * posPtr    = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * triPtr    = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    const int32_t * rangesPtr = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    if (d.instanceMode) { rangesPtr = nullptr; }
    // - Outputs
    float * outPtrs[2];
    outPtrs[0] = reinterpret_cast<float *>(buffers[iBuf++]);
    outPtrs[1] = reinterpret_cast<float *>(buffers[iBuf++]);
    // if (!d.enableDB) outPtrs[1] = nullptr;
    // assert(d.enableDB);  // In torch impl, it's always enabled.

    // Set up CudaRaster.
    cr->setViewportSize(d.width, d.height, d.depth);
    cr->setVertexBuffer((void*)posPtr, d.posCount);
    cr->setIndexBuffer ((void*)triPtr, d.triCount);

    bool enablePeel = false; // TODO: support peeling_idx
    cr->setRenderModeFlags(enablePeel ? CR::CudaRaster::RenderModeFlag_EnableDepthPeeling : 0);  // No backface culling.
    if (enablePeel) {
        cr->swapDepthAndPeel(); // Use previous depth buffer as peeling depth input.
    }

    // Run CudaRaster in one large batch. In case of error, the workload could be split into smaller batches - maybe do that in the future.
    cr->deferredClear(0u);
    bool success = cr->drawTriangles(rangesPtr, stream);
    NVDR_CHECK(success, "subtriangle count overflow");

    // Populate pixel shader kernel parameters.
    RasterizeCudaFwdShaderParams p;
    p.pos = posPtr;
    p.tri = triPtr;
    p.in_idx = (const int*)cr->getColorBuffer();
    p.out = outPtrs[0];
    p.out_db = outPtrs[1];
    p.numTriangles = triCount;
    p.numVertices = posCount;
    p.width  = width;
    p.height = height;
    p.depth  = depth;
    p.instance_mode = (pos.sizes().size() > 2) ? 1 : 0;
    p.xs = 2.f / (float)p.width;
    p.xo = 1.f / (float)p.width - 1.f;
    p.ys = 2.f / (float)p.height;
    p.yo = 1.f / (float)p.height - 1.f;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos & 15),    "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.out & 15),    "out output tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.out_db & 15), "out_db output tensor not aligned to float4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_WIDTH, RAST_CUDA_FWD_SHADER_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)RasterizeCudaFwdShaderKernel, gridSize, blockSize, args, 0, stream));
}

void rasterizeBwd(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    // Get descriptor
    auto const & d = *UnpackDescriptor<RasterizeDescriptor>(opaque, opaque_len);

    // All buffers.
    size_t iBuf = 0;
    // Inputs
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