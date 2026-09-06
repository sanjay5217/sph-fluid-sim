#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

constant float kPointSize = 8.0;
constant float kCircleSoftEdgeInner = 0.4;
constant float kCircleSoftEdgeOuter = 0.5;
constant float3 kParticleColor = float3(0.2, 0.6, 1.0);
constant float kBoundaryRestitution = 0.8;
constant float kRestVelocityEpsilon = 0.2;

struct VertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
};

vertex VertexOut particleVertex(uint instanceID [[instance_id]], constant Particle *particles [[buffer(0)]]) {
    VertexOut out;
    float2 worldPos = particles[instanceID].position;
    float2 ndcPos = worldPos * float2(kMetersToNDCX, kMetersToNDCY);
    out.position = float4(ndcPos, 0.0, 1.0);
    out.pointSize = kPointSize;
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

kernel void updateParticlePosition(
            device Particle *particles [[buffer(0)]],
            const constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    Particle p = particles[id];
    p.velocity.y += params.gravity * params.dt;
    p.position += p.velocity * params.dt;

    if (p.position.x > kContainerHalfWidth) {
        p.position.x = kContainerHalfWidth;
        p.velocity.x = (abs(p.velocity.x) < kRestVelocityEpsilon) ? 0.0 : -p.velocity.x * kBoundaryRestitution;
    } else if (p.position.x < -kContainerHalfWidth) {
        p.position.x = -kContainerHalfWidth;
        p.velocity.x = (abs(p.velocity.x) < kRestVelocityEpsilon) ? 0.0 : -p.velocity.x * kBoundaryRestitution;
    }

    if (p.position.y > kContainerHalfHeight) {
        p.position.y = kContainerHalfHeight;
        p.velocity.y = (abs(p.velocity.y) < kRestVelocityEpsilon) ? 0.0 : -p.velocity.y * kBoundaryRestitution;
    } else if (p.position.y < -kContainerHalfHeight) {
        p.position.y = -kContainerHalfHeight;
        p.velocity.y = (abs(p.velocity.y) < kRestVelocityEpsilon) ? 0.0 : -p.velocity.y * kBoundaryRestitution;
    }

    particles[id] = p;
}

float poly6(float r, float h)
{
    if (r >= h) {
        return 0.0;
    }

    float h2 = h * h;
    float diff = h2 - r * r;

    return (315.0f / (64.0f * M_PI_F * pow(h, 9.0f))) * diff * diff * diff;
}

kernel void updateDensity(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    float2 position = particles[id].position;
    float density = 0.0;
    
    // Current neighbor search is O(n) per particle
    for (uint j = 0; j < params.particleCount; j++) {
        float r = length(position - particles[j].position);
        density += params.mass * poly6(r, params.smoothingRadius);
    }

    particles[id].density = density;
}
