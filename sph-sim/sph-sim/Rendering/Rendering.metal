#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

constant float kCircleSoftEdgeInner = 0.4;
constant float kCircleSoftEdgeOuter = 0.5;
constant float3 kParticleColor = float3(0.2, 0.6, 1.0);

struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
};

vertex VertexOut particleVertex(uint instanceID [[instance_id]],
                                constant Particle *particles [[buffer(0)]],
                                constant SimulationParams& params [[buffer(1)]]) {
    VertexOut out;
    float2 worldPos = particles[instanceID].position;
    float2 ndcPos = worldPos * float2(kMetersToNDCX, kMetersToNDCY);
    out.position = float4(ndcPos, 0.0, 1.0);
    out.pointSize = params.pointSize;
    return out;
}

fragment float4 particleFragment(float2 pointCoord [[point_coord]]) {
    float dist = length(pointCoord - float2(0.5));
    float alpha = 1.0 - smoothstep(kCircleSoftEdgeInner, kCircleSoftEdgeOuter, dist);
    if (alpha <= 0.0) {
        discard_fragment();
    }
    return float4(kParticleColor, alpha);
}
