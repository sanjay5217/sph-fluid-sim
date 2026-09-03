#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
};

vertex VertexOut particleVertex(uint instanceID [[instance_id]],
                                 constant Particle *particles [[buffer(0)]]) {
    VertexOut out;
    float2 pos = particles[instanceID].position;
    out.position = float4(pos, 0.0, 1.0);
    out.pointSize = 8.0;
    return out;
}

fragment float4 particleFragment() {
    return float4(0.2, 0.6, 1.0, 1.0);
}
