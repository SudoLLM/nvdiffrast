#pragma once
#include <stdint.h>
#include "../common/common.h"
#include "../common/glutil.h"
#include "../common/glutil_extlist.h"

struct GemotryDescriptor {
    int32_t W, H;
    int32_t N, V;
    int32_t F;
};

void rasterize_fwd(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);