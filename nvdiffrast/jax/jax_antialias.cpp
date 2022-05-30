#include <cuda.h>
#include "jax_antialias.h"
#include "kernel_helpers.h"
#include "../common/common.h"
#include "../common/framework.h"
#include "../common/antialias.h"

//------------------------------------------------------------------------
// Kernel prototypes.

void AntialiasFwdMeshKernel         (const AntialiasKernelParams p);
void AntialiasFwdDiscontinuityKernel(const AntialiasKernelParams p);
void AntialiasFwdAnalysisKernel     (const AntialiasKernelParams p);
void AntialiasGradKernel            (const AntialiasKernelParams p);

//------------------------------------------------------------------------
// Forward

void antialiasFwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    // Get descriptor
    auto const & d = *UnpackDescriptor<AntialiasDescriptor>(opaque, opaque_len);

    AntialiasKernelParams p = {}; // Initialize all fields to zero.
    memset(&p, 0, sizeof(p));  // TODO: necessary?

    // All inputs and outputs are given in buffers
    // - Inputs
    size_t iBuf = 0;
    const float   * colorPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * rastPtr  = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * posPtr   = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * triPtr   = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    // - Outputs
    float   * outPtr = reinterpret_cast<float *>(buffers[iBuf++]); // out
    int32_t * bufPtr = reinterpret_cast<int32_t *>(buffers[iBuf++]); // out_da

    // Instance rendering mode?
    p.instance_mode = d.instanceMode;

    // Extract input dimensions.
    p.numVertices  = d.numVertices;
    p.numTriangles = d.numTriangles;
    p.n            = d.n;
    p.height       = d.height;
    p.width        = d.width;
    p.channels     = d.channels;

    // Sanity checks in python code

    // Misc parameters.
    p.xh = .5f * (float)p.width;
    p.yh = .5f * (float)p.height;

    // Get input pointers.
    p.color = colorPtr;
    p.rasterOut = rastPtr;
    p.pos = posPtr;
    p.tri = triPtr;

    // Output tensor.
    p.output = outPtr;
    p.workBuffer = (int4*)(bufPtr);  // Work buffer. One extra int4 for storing counters.

    // Clear the work counters.
    NVDR_CHECK_CUDA_ERROR(cudaMemsetAsync(p.workBuffer, 0, sizeof(int4), stream));

    // Kernel parameters.
    void* args[] = {&p};

    // TODO: cache evHash?
    // Calculate opposite vertex hash.
    {
        if (p.allocTriangles < p.numTriangles) {
            p.allocTriangles = std::max(p.allocTriangles, 64);
            while (p.allocTriangles < p.numTriangles) {
                p.allocTriangles <<= 1; // Must be power of two.
            }

            // (Re-)allocate memory for the hash.
            NVDR_CHECK_CUDA_ERROR(cudaMalloc(&p.evHash, p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE * sizeof(uint4)));
            LOG(INFO) << "Increasing topology hash size to accommodate " << p.allocTriangles << " triangles";
        }

        // Clear the hash and launch the mesh kernel to populate it.
        NVDR_CHECK_CUDA_ERROR(cudaMemsetAsync(p.evHash, 0, p.allocTriangles * AA_HASH_ELEMENTS_PER_TRIANGLE * sizeof(uint4), stream));
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasFwdMeshKernel, (p.numTriangles - 1) / AA_MESH_KERNEL_THREADS_PER_BLOCK + 1, AA_MESH_KERNEL_THREADS_PER_BLOCK, args, 0, stream));
    }

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos        & 15), "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rasterOut  &  7), "raster_out input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p.workBuffer & 15), "work_buffer internal tensor not aligned to int4");
    NVDR_CHECK(!((uintptr_t)p.evHash     & 15), "topology_hash internal tensor not aligned to int4");

    // Copy input to output as a baseline.
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(p.output, p.color, p.n * p.height * p.width * p.channels * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Choose launch parameters for the discontinuity finder kernel and launch.
    dim3 blockSize(AA_DISCONTINUITY_KERNEL_BLOCK_WIDTH, AA_DISCONTINUITY_KERNEL_BLOCK_HEIGHT, 1);
    dim3 gridSize = getLaunchGridSize(blockSize, p.width, p.height, p.n);
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasFwdDiscontinuityKernel, gridSize, blockSize, args, 0, stream));

    // Determine optimum block size for the persistent analysis kernel.
    int device = 0;
    int numCTA = 0;
    int numSM  = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGetDevice(&device));
    NVDR_CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numCTA, (void*)AntialiasFwdAnalysisKernel, AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK, 0));
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device));
    // Launch analysis kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasFwdAnalysisKernel, numCTA * numSM, AA_ANALYSIS_KERNEL_THREADS_PER_BLOCK, args, 0, stream));

    // TODO: cache?
    // Release evHash
    NVDR_CHECK_CUDA_ERROR(cudaFree(p.evHash));
}

//------------------------------------------------------------------------
// Backward

void antialiasBwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    // Get descriptor
    auto const & d = *UnpackDescriptor<AntialiasDescriptor>(opaque, opaque_len);

    AntialiasKernelParams p = {}; // Initialize all fields to zero.
    memset(&p, 0, sizeof(p));  // TODO: necessary?

    // All inputs and outputs are given in buffers
    // - Inputs
    size_t iBuf = 0;
    const float   * colorPtr = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * rastPtr  = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const float   * posPtr   = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * triPtr   = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    const float   * dyPtr    = reinterpret_cast<const float   *>(buffers[iBuf++]);
    const int32_t * bufPtr   = reinterpret_cast<const int32_t *>(buffers[iBuf++]);
    // - Outputs
    float * gradColPtr = reinterpret_cast<float *>(buffers[iBuf++]);
    float * gradPosPtr = reinterpret_cast<float *>(buffers[iBuf++]);

    // Instance rendering mode?
    p.instance_mode = d.instanceMode;

    // Extract input dimensions.
    p.numVertices  = d.numVertices;
    p.numTriangles = d.numTriangles;
    p.n            = d.n;
    p.height       = d.height;
    p.width        = d.width;
    p.channels     = d.channels;

    // Misc parameters.
    p.xh = .5f * (float)p.width;
    p.yh = .5f * (float)p.height;
    
    // Sanity checks in python code

    // Get input pointers.
    p.color = colorPtr;
    p.rasterOut = rastPtr;
    p.tri = triPtr;
    p.pos = posPtr;
    p.dy = dyPtr;
    p.workBuffer = (int4*)(bufPtr);

    // Get output pointers.
    p.gradColor = gradColPtr;
    p.gradPos = gradPosPtr;

    // Initialize all the stuff.
    NVDR_CHECK_CUDA_ERROR(cudaMemsetAsync(&p.workBuffer[0].y, 0, sizeof(int), stream)); // Gradient kernel work counter.
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(p.gradColor, p.dy, p.n * p.height * p.width * p.channels * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    NVDR_CHECK_CUDA_ERROR(cudaMemsetAsync(p.gradPos, 0, (p.instance_mode ? p.n : 1) * p.numVertices * 4 * sizeof(float), stream));

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos        & 15), "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.workBuffer & 15), "work_buffer internal tensor not aligned to int4");

    // Determine optimum block size for the gradient kernel and launch.
    void* args[] = {&p};
    int device = 0;
    int numCTA = 0;
    int numSM  = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGetDevice(&device));
    NVDR_CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numCTA, (void*)AntialiasGradKernel, AA_GRAD_KERNEL_THREADS_PER_BLOCK, 0));
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device));
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)AntialiasGradKernel, numCTA * numSM, AA_GRAD_KERNEL_THREADS_PER_BLOCK, args, 0, stream));
}
