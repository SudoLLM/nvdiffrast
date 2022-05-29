#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

struct RasterizeDescriptor {
    int  width;
    int  height;
    bool enableDB;
    bool instanceMode;
    int  posCount;
    int  triCount;
    int  vtxPerInstance;
    int  depth;
};

void rasterize_fwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);