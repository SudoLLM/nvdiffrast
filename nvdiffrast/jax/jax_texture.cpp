#include <cuda.h>
#include "jax_texture.h"
#include "kernel_helpers.h"
#include "../common/common.h"
#include "../common/framework.h"
#include "../common/texture.h"


//------------------------------------------------------------------------
// Kernel prototypes.

void MipBuildKernel1                            (const TextureKernelParams p);
void MipBuildKernel2                            (const TextureKernelParams p);
void MipBuildKernel4                            (const TextureKernelParams p);
void TextureFwdKernelNearest1                   (const TextureKernelParams p);
void TextureFwdKernelNearest2                   (const TextureKernelParams p);
void TextureFwdKernelNearest4                   (const TextureKernelParams p);
void TextureFwdKernelLinear1                    (const TextureKernelParams p);
void TextureFwdKernelLinear2                    (const TextureKernelParams p);
void TextureFwdKernelLinear4                    (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearest1       (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearest2       (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearest4       (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinear1        (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinear2        (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinear4        (const TextureKernelParams p);
void TextureFwdKernelCubeNearest1               (const TextureKernelParams p);
void TextureFwdKernelCubeNearest2               (const TextureKernelParams p);
void TextureFwdKernelCubeNearest4               (const TextureKernelParams p);
void TextureFwdKernelCubeLinear1                (const TextureKernelParams p);
void TextureFwdKernelCubeLinear2                (const TextureKernelParams p);
void TextureFwdKernelCubeLinear4                (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearest1   (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearest2   (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearest4   (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinear1    (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinear2    (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinear4    (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearestBO1     (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearestBO2     (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapNearestBO4     (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinearBO1      (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinearBO2      (const TextureKernelParams p);
void TextureFwdKernelLinearMipmapLinearBO4      (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearestBO1 (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearestBO2 (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapNearestBO4 (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinearBO1  (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinearBO2  (const TextureKernelParams p);
void TextureFwdKernelCubeLinearMipmapLinearBO4  (const TextureKernelParams p);
void MipGradKernel1                             (const TextureKernelParams p);
void MipGradKernel2                             (const TextureKernelParams p);
void MipGradKernel4                             (const TextureKernelParams p);
void TextureGradKernelNearest                   (const TextureKernelParams p);
void TextureGradKernelLinear                    (const TextureKernelParams p);
void TextureGradKernelLinearMipmapNearest       (const TextureKernelParams p);
void TextureGradKernelLinearMipmapLinear        (const TextureKernelParams p);
void TextureGradKernelCubeNearest               (const TextureKernelParams p);
void TextureGradKernelCubeLinear                (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapNearest   (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapLinear    (const TextureKernelParams p);
void TextureGradKernelLinearMipmapNearestBO     (const TextureKernelParams p);
void TextureGradKernelLinearMipmapLinearBO      (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapNearestBO (const TextureKernelParams p);
void TextureGradKernelCubeLinearMipmapLinearBO  (const TextureKernelParams p);

//------------------------------------------------------------------------
// Modeselektor.

static void set_modes(TextureKernelParams& p, int filter_mode, int boundary_mode, int max_mip_level) {
    // Mip and filter modes.
    p.filterMode = filter_mode;
    NVDR_CHECK(p.filterMode >= 0 && p.filterMode < TEX_MODE_COUNT, "filter_mode unsupported");
    p.enableMip = (p.filterMode == TEX_MODE_LINEAR_MIPMAP_NEAREST || p.filterMode == TEX_MODE_LINEAR_MIPMAP_LINEAR);

    // Mip level clamp.
    if (p.enableMip) {
        p.mipLevelLimit = max_mip_level;
        NVDR_CHECK(p.mipLevelLimit >= -1, "invalid max_mip_level");
    }

    // Boundary mode.
    p.boundaryMode = boundary_mode;
    NVDR_CHECK(p.boundaryMode >= 0 && p.boundaryMode < TEX_BOUNDARY_MODE_COUNT, "boundary_mode unsupported");
}

void textureFwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    // All inputs and outputs are given in buffers.
    size_t iBuf = 0;
    // - Inputs
    const float * texPtr = reinterpret_cast<const float *>(buffers[iBuf++]);
    const float * uvPtr  = reinterpret_cast<const float *>(buffers[iBuf++]);
    // - Outputs
    float * outPtr = reinterpret_cast<float *>(buffers[iBuf++]);

    // Get descriptor
    auto const & d = *UnpackDescriptor<TextureDescriptor>(opaque, opaque_len);

    TextureKernelParams p = {}; // Initialize all fields to zero.
    bool has_mip_stack = false;
    int max_mip_level = 0; // TODO: not sure.
    set_modes(p, d.filterMode, d.boundaryMode, max_mip_level);

    // See if we have these tensors or not.
    bool has_uv_da = false;
    bool has_mip_level_bias = false;
    // if (p.enableMip) {
    //     NVDR_CHECK(has_uv_da || has_mip_level_bias, "mipmapping filter mode requires uv_da and/or mip_level_bias input");
    //     NVDR_CHECK(has_mip_stack || mip_w.defined(), "mipmapping filter mode requires mip wrapper or mip stack input");
    // }
    
    bool cube_mode = false;
    if (!cube_mode) {
        p.texHeight = d.texHeight;
        p.texWidth = d.texWidth;
        p.channels = d.texChannels;
    }
    else {
        // TODO:
    }
    
    p.n = d.uvBatchSize;
    p.imgHeight = d.uvHeight;
    p.imgWidth = d.uvWidth;
    p.texDepth = d.texBatchSize;

    p.tex[0] = texPtr;
    p.uv = uvPtr;
    p.uvDA = nullptr;
    p.mipLevelBias = nullptr;

    p.out = outPtr;
    
    // Choose kernel variants based on channel count.
    void* args[] = {&p};
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Mip-related setup.
    float* pmip = 0;

    // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
    if (!cube_mode)
        NVDR_CHECK(!((uintptr_t)p.uv & 7), "uv input tensor not aligned to float2");
    if ((p.channels & 3) == 0) {
        for (int i=0; i <= p.mipLevelMax; i++)
            NVDR_CHECK(!((uintptr_t)p.tex[i] & 15), "tex or mip input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.out    & 15), "out output tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)pmip     & 15), "mip input tensor not aligned to float4");
    }
    if ((p.channels & 1) == 0) {
        for (int i=0; i <= p.mipLevelMax; i++)
            NVDR_CHECK(!((uintptr_t)p.tex[i] & 7), "tex or mip input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.out    & 7), "out output tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)pmip     & 7), "mip input tensor not aligned to float2");
    }
    if (!cube_mode) {
        NVDR_CHECK(!((uintptr_t)p.uvDA & 15), "uv_da input tensor not aligned to float4");
    }
    else {
        NVDR_CHECK(!((uintptr_t)p.uvDA & 7), "uv_da input tensor not aligned to float2");
    }

    // Choose launch parameters for texture lookup kernel.
    dim3 blockSize = getLaunchBlockSize(TEX_FWD_MAX_KERNEL_BLOCK_WIDTH, TEX_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    // Choose kernel based on filter mode, cube mode, bias-only mode, and datatype.
    void* func_tbl[TEX_MODE_COUNT * 2 * 2 * 3] = {
        (void*)TextureFwdKernelNearest1,
        (void*)TextureFwdKernelNearest2,
        (void*)TextureFwdKernelNearest4,
        (void*)TextureFwdKernelLinear1,
        (void*)TextureFwdKernelLinear2,
        (void*)TextureFwdKernelLinear4,
        (void*)TextureFwdKernelLinearMipmapNearest1,
        (void*)TextureFwdKernelLinearMipmapNearest2,
        (void*)TextureFwdKernelLinearMipmapNearest4,
        (void*)TextureFwdKernelLinearMipmapLinear1,
        (void*)TextureFwdKernelLinearMipmapLinear2,
        (void*)TextureFwdKernelLinearMipmapLinear4,
        (void*)TextureFwdKernelCubeNearest1,
        (void*)TextureFwdKernelCubeNearest2,
        (void*)TextureFwdKernelCubeNearest4,
        (void*)TextureFwdKernelCubeLinear1,
        (void*)TextureFwdKernelCubeLinear2,
        (void*)TextureFwdKernelCubeLinear4,
        (void*)TextureFwdKernelCubeLinearMipmapNearest1,
        (void*)TextureFwdKernelCubeLinearMipmapNearest2,
        (void*)TextureFwdKernelCubeLinearMipmapNearest4,
        (void*)TextureFwdKernelCubeLinearMipmapLinear1,
        (void*)TextureFwdKernelCubeLinearMipmapLinear2,
        (void*)TextureFwdKernelCubeLinearMipmapLinear4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        (void*)TextureFwdKernelLinearMipmapNearestBO1,
        (void*)TextureFwdKernelLinearMipmapNearestBO2,
        (void*)TextureFwdKernelLinearMipmapNearestBO4,
        (void*)TextureFwdKernelLinearMipmapLinearBO1,
        (void*)TextureFwdKernelLinearMipmapLinearBO2,
        (void*)TextureFwdKernelLinearMipmapLinearBO4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        (void*)TextureFwdKernelCubeLinearMipmapNearestBO1,
        (void*)TextureFwdKernelCubeLinearMipmapNearestBO2,
        (void*)TextureFwdKernelCubeLinearMipmapNearestBO4,
        (void*)TextureFwdKernelCubeLinearMipmapLinearBO1,
        (void*)TextureFwdKernelCubeLinearMipmapLinearBO2,
        (void*)TextureFwdKernelCubeLinearMipmapLinearBO4,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (cube_mode)
        func_idx += TEX_MODE_COUNT; // Cube variant.
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT * 2; // Bias-only variant.
    func_idx = func_idx * 3 + channel_div_idx; // Choose vector size.

    // Launch kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));
}

void textureLinearBwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    // All buffers.
    size_t iBuf = 0;
    // Inputs
    const float * texPtr = reinterpret_cast<const float *>(buffers[iBuf++]);
    const float * uvPtr  = reinterpret_cast<const float *>(buffers[iBuf++]);
    const float * dyPtr  = reinterpret_cast<const float *>(buffers[iBuf++]);
    // Outputs
    float * gradTexPtr = reinterpret_cast<float *>(buffers[iBuf++]);
    float * gradUvPtr  = reinterpret_cast<float *>(buffers[iBuf++]);

    // Get descriptor
    auto const & d = *UnpackDescriptor<TextureDescriptor>(opaque, opaque_len);

    TextureKernelParams p = {}; // Initialize all fields to zero.
    bool has_mip_stack = false;
    int max_mip_level = 0;  // TODO: ?
    set_modes(p, d.filterMode, d.boundaryMode, max_mip_level);

    // See if we have these tensors or not.
    bool has_uv_da = false;
    bool has_mip_level_bias = false;

    // Sanity checks and state setters.
    bool cube_mode = false;
    if (!cube_mode) {
        p.texHeight = d.texHeight;
        p.texWidth  = d.texWidth;
        p.channels  = d.texChannels;
    }
    else {
        // TODO:
    }
    p.n = d.uvBatchSize;
    p.imgHeight = d.uvHeight;
    p.imgWidth = d.uvWidth;
    p.texDepth = d.texBatchSize;

    // Get input pointers.
    p.tex[0] = texPtr;
    p.uv = uvPtr;
    p.dy = dyPtr;
    p.uvDA = nullptr;
    p.mipLevelBias = nullptr;

    // Allocate output tensor for tex gradient.
    size_t gradTexBytes = d.texBatchSize * d.texHeight * d.texWidth * d.texChannels * sizeof(float);
    cudaMemsetAsync(gradTexPtr, 0, gradTexBytes, stream);
    p.gradTex[0] = gradTexPtr;

    // Allocate output tensor for uv gradient.
    if (p.filterMode != TEX_MODE_NEAREST) {
        p.gradUV = gradUvPtr;
    }

    // Choose kernel variants based on channel count.
    int channel_div_idx = 0;
    if (!(p.channels & 3))
        channel_div_idx = 2;  // Channel count divisible by 4.
    else if (!(p.channels & 1))
        channel_div_idx = 1;  // Channel count divisible by 2.

    // Mip-related setup.
    float* pmip = 0;
    float* pgradMip = 0;

    // Verify that buffers are aligned to allow float2/float4 operations. Unused pointers are zero so always aligned.
    if (!cube_mode) {
        NVDR_CHECK(!((uintptr_t)p.uv       & 7), "uv input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.gradUV   & 7), "grad_uv output tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.uvDA     & 15), "uv_da input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)p.gradUVDA & 15), "grad_uv_da output tensor not aligned to float4");
    }
    else {
        NVDR_CHECK(!((uintptr_t)p.uvDA     & 7), "uv_da input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)p.gradUVDA & 7), "grad_uv_da output tensor not aligned to float2");
    }
    if ((p.channels & 3) == 0) {
        for (int i=0; i <= p.mipLevelMax; i++) {
            NVDR_CHECK(!((uintptr_t)p.tex[i]     & 15), "tex or mip input tensor not aligned to float4");
            NVDR_CHECK(!((uintptr_t)p.gradTex[i] & 15), "grad_tex output tensor not aligned to float4");
        }
        NVDR_CHECK(!((uintptr_t)p.dy         & 15), "dy input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)pmip         & 15), "mip input tensor not aligned to float4");
        NVDR_CHECK(!((uintptr_t)pgradMip     & 15), "internal mip gradient tensor not aligned to float4");
    }
    if ((p.channels & 1) == 0) {
        for (int i=0; i <= p.mipLevelMax; i++) {
            NVDR_CHECK(!((uintptr_t)p.tex[i]     & 7), "tex or mip input tensor not aligned to float2");
            NVDR_CHECK(!((uintptr_t)p.gradTex[i] & 7), "grad_tex output tensor not aligned to float2");
        }
        NVDR_CHECK(!((uintptr_t)p.dy         & 7), "dy output tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)pmip         & 7), "mip input tensor not aligned to float2");
        NVDR_CHECK(!((uintptr_t)pgradMip     & 7), "internal mip gradient tensor not aligned to float2");
    }

    // Choose launch parameters for main gradient kernel.
    void* args[] = {&p};
    dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.imgWidth, p.imgHeight);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.imgWidth, p.imgHeight, p.n);

    void* func_tbl[TEX_MODE_COUNT * 2 * 2] = {
        (void*)TextureGradKernelNearest,
        (void*)TextureGradKernelLinear,
        (void*)TextureGradKernelLinearMipmapNearest,
        (void*)TextureGradKernelLinearMipmapLinear,
        (void*)TextureGradKernelCubeNearest,
        (void*)TextureGradKernelCubeLinear,
        (void*)TextureGradKernelCubeLinearMipmapNearest,
        (void*)TextureGradKernelCubeLinearMipmapLinear,
        NULL,
        NULL,
        (void*)TextureGradKernelLinearMipmapNearestBO,
        (void*)TextureGradKernelLinearMipmapLinearBO,
        NULL,
        NULL,
        (void*)TextureGradKernelCubeLinearMipmapNearestBO,
        (void*)TextureGradKernelCubeLinearMipmapLinearBO,
    };

    // Function index.
    int func_idx = p.filterMode;
    if (cube_mode)
        func_idx += TEX_MODE_COUNT; // Cube variant.
    if (p.enableMip && !has_uv_da)
        func_idx += TEX_MODE_COUNT * 2; // Bias-only variant.

    // Launch main gradient kernel.
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream));

    // Launch kernel to pull gradients from mip levels. Don't do this if mip stack was supplied - individual level gradients are already there.
    if (p.enableMip && !has_mip_stack) {
        dim3 blockSize = getLaunchBlockSize(TEX_GRAD_MAX_MIP_KERNEL_BLOCK_WIDTH, TEX_GRAD_MAX_MIP_KERNEL_BLOCK_HEIGHT, p.texWidth, p.texHeight);
        dim3 gridSize  = getLaunchGridSize(blockSize, p.texWidth, p.texHeight, p.texDepth * (cube_mode ? 6 : 1));
        int sharedBytes = blockSize.x * blockSize.y * p.channels * sizeof(float);

        void* mip_grad_func_tbl[3] = { (void*)MipGradKernel1, (void*)MipGradKernel2, (void*)MipGradKernel4 };
        NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(mip_grad_func_tbl[channel_div_idx], gridSize, blockSize, args, sharedBytes, stream));
    }
}