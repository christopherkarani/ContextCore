#include <metal_stdlib>
using namespace metal;

constant uint CENTRALITY_TILE_SIZE = 16;
constant uint CENTRALITY_DIM_TILE = 32;

kernel void token_centrality(
    device const float* embeddings   [[buffer(0)]],
    device float* centrality         [[buffer(1)]],
    constant uint& dim               [[buffer(2)]],
    constant uint& n                 [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tgs                         [[threads_per_threadgroup]]
) {
    if (gid >= n) {
        return;
    }

    if (n <= 1) {
        centrality[gid] = 0.0f;
        return;
    }

    const uint selfBase = gid * dim;

    float normSelfSq = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        const float v = embeddings[selfBase + d];
        normSelfSq += v * v;
    }
    const float normSelf = sqrt(normSelfSq);

    threadgroup float shared_tile[CENTRALITY_TILE_SIZE * CENTRALITY_DIM_TILE];

    float sumSimilarity = 0.0f;

    for (uint jBase = 0; jBase < n; jBase += CENTRALITY_TILE_SIZE) {
        const uint tileCount = min(CENTRALITY_TILE_SIZE, n - jBase);

        float dotAcc[CENTRALITY_TILE_SIZE];
        float normAcc[CENTRALITY_TILE_SIZE];

        for (uint t = 0; t < tileCount; ++t) {
            dotAcc[t] = 0.0f;
            normAcc[t] = 0.0f;
        }

        for (uint dBase = 0; dBase < dim; dBase += CENTRALITY_DIM_TILE) {
            const uint dCount = min(CENTRALITY_DIM_TILE, dim - dBase);
            const uint loadCount = tileCount * dCount;

            for (uint loadIndex = tid; loadIndex < loadCount; loadIndex += tgs) {
                const uint localJ = loadIndex / dCount;
                const uint localD = loadIndex % dCount;
                const uint globalJ = jBase + localJ;
                shared_tile[localJ * CENTRALITY_DIM_TILE + localD] = embeddings[globalJ * dim + dBase + localD];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint localJ = 0; localJ < tileCount; ++localJ) {
                const uint globalJ = jBase + localJ;
                if (globalJ == gid) {
                    continue;
                }

                float partialDot = 0.0f;
                float partialNorm = 0.0f;

                for (uint localD = 0; localD < dCount; ++localD) {
                    const float other = shared_tile[localJ * CENTRALITY_DIM_TILE + localD];
                    const float self = embeddings[selfBase + dBase + localD];
                    partialDot += self * other;
                    partialNorm += other * other;
                }

                dotAcc[localJ] += partialDot;
                normAcc[localJ] += partialNorm;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint localJ = 0; localJ < tileCount; ++localJ) {
            const uint globalJ = jBase + localJ;
            if (globalJ == gid) {
                continue;
            }

            const float denom = normSelf * sqrt(normAcc[localJ]);
            if (denom > 0.0f) {
                sumSimilarity += dotAcc[localJ] / denom;
            }
        }
    }

    centrality[gid] = sumSimilarity / float(n - 1);
}

kernel void cross_attention_score(
    device const float* taskQuery      [[buffer(0)]],
    device const float* embeddings     [[buffer(1)]],
    device const float* centrality     [[buffer(2)]],
    device const float2* weights       [[buffer(3)]],
    device float* evictionScores       [[buffer(4)]],
    constant uint& dim                 [[buffer(5)]],
    constant uint& n                   [[buffer(6)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }

    const uint base = gid * dim;

    float dot = 0.0f;
    float queryNormSq = 0.0f;
    float chunkNormSq = 0.0f;

    for (uint d = 0; d < dim; ++d) {
        const float q = taskQuery[d];
        const float c = embeddings[base + d];
        dot += q * c;
        queryNormSq += q * q;
        chunkNormSq += c * c;
    }

    const float denom = sqrt(queryNormSq) * sqrt(chunkNormSq);
    float relevance = 0.0f;
    if (denom > 0.0f) {
        relevance = dot / denom;
    }

    evictionScores[gid] = relevance * weights[0].x + centrality[gid] * weights[0].y;
}
