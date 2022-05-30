#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

struct AntialiasDescriptor {
    int  numVertices;
    int  numTriangles;
    int  n;
    int  width;
    int  height;
    int  channels;
    bool instanceMode;
};

void antialiasFwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
void antialiasBwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);