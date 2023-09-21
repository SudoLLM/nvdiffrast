#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

struct TextureDescriptor {
    int filterMode;
    int boundaryMode;
    int texBatchSize;
    int texHeight;
    int texWidth;
    int texChannels;
    int uvBatchSize;
    int uvHeight;
    int uvWidth;
};

void textureFwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
void textureLinearBwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
