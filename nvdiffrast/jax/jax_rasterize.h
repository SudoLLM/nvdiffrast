#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

struct GemotryDescriptor {
    int32_t W, H;
    int32_t N, V;
    int32_t F;
};

void rasterize_fwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);