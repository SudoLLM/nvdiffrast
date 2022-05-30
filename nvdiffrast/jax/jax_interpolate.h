#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <vector>

struct InterpolateDescriptor {
    int  numVertices;
    int  numTriangles;
    int  numAttr;
    int  attrBC;
    int  attrDepth;
    int  width;
    int  height;
    int  depth;
    bool enableDB;
    bool instanceMode;
    bool diffAttrsAll;
    std::vector<int> diffAttrsVec;
};

void interpolateFwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);
void interpolateBwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);