#include <metal_stdlib>
using namespace metal;

kernel void relevance_score(
    device const float* query        [[buffer(0)]],
    device const float* chunks       [[buffer(1)]],
    device const float* recencyWts   [[buffer(2)]],
    constant float2& weights         [[buffer(3)]],
    device float* scores             [[buffer(4)]],
    constant uint& dim               [[buffer(5)]],
    constant uint& n                 [[buffer(6)]],
    constant float& queryNorm        [[buffer(7)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }

    const uint base = gid * dim;

    float dot = 0.0f;
    float chunkNormSq = 0.0f;

    for (uint i = 0; i < dim; ++i) {
        const float q = query[i];
        const float c = chunks[base + i];
        dot += q * c;
        chunkNormSq += c * c;
    }

    const float chunkNorm = sqrt(chunkNormSq);

    float cosine = 0.0f;
    const float denom = queryNorm * chunkNorm;
    if (denom > 0.0f) {
        cosine = dot / denom;
    }

    scores[gid] = cosine * weights.x + recencyWts[gid] * weights.y;
}

kernel void topk_indices(
    device const float* scores   [[buffer(0)]],
    device uint* indices         [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    constant uint& k             [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }

    if (n == 0 || k == 0) {
        return;
    }

    const uint outCount = min(n, k);

    for (uint rank = 0; rank < outCount; ++rank) {
        float bestScore = -INFINITY;
        uint bestIndex = 0;

        for (uint i = 0; i < n; ++i) {
            bool selected = false;
            for (uint prev = 0; prev < rank; ++prev) {
                if (indices[prev] == i) {
                    selected = true;
                    break;
                }
            }

            if (selected) {
                continue;
            }

            const float score = scores[i];
            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }

        indices[rank] = bestIndex;
    }

    for (uint rank = outCount; rank < k; ++rank) {
        indices[rank] = 0;
    }
}
