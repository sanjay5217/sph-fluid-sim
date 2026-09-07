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

// Soft wall
float2 softWallAcceleration(float distance, float2 normal, float2 velocity,
                            const constant SimulationParams& params) {
    float overlap = params.wallBand - distance;
    if (overlap <= 0.0) {
        return float2(0.0);
    }

    float approach = min(dot(velocity, normal), 0.0f);
    return (params.wallStiffness * overlap - params.wallDamping * approach) * normal;
}

kernel void updateParticlePosition(
            device Particle *particles [[buffer(0)]],
            const constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    Particle p = particles[id];

    float2 wallAcceleration =
          softWallAcceleration(kContainerHalfWidth  - p.position.x, float2(-1.0,  0.0), p.velocity, params)
        + softWallAcceleration(kContainerHalfWidth  + p.position.x, float2( 1.0,  0.0), p.velocity, params)
        + softWallAcceleration(kContainerHalfHeight - p.position.y, float2( 0.0, -1.0), p.velocity, params)
        + softWallAcceleration(kContainerHalfHeight + p.position.y, float2( 0.0,  1.0), p.velocity, params);

    float2 acceleration = p.force / params.mass + wallAcceleration;
    p.velocity += acceleration * params.dt;
    p.position += p.velocity * params.dt;

    // Backstop clamp
    if (p.position.x > kContainerHalfWidth) {
        p.position.x = kContainerHalfWidth;
        p.velocity.x = min(p.velocity.x, 0.0f);
    } else if (p.position.x < -kContainerHalfWidth) {
        p.position.x = -kContainerHalfWidth;
        p.velocity.x = max(p.velocity.x, 0.0f);
    }

    if (p.position.y > kContainerHalfHeight) {
        p.position.y = kContainerHalfHeight;
        p.velocity.y = min(p.velocity.y, 0.0f);
    } else if (p.position.y < -kContainerHalfHeight) {
        p.position.y = -kContainerHalfHeight;
        p.velocity.y = max(p.velocity.y, 0.0f);
    }

    particles[id] = p;
}

// poly6
float poly6(float r, float h)
{
    if (r >= h) {
        return 0.0;
    }

    float h2 = h * h;
    float diff = h2 - r * r;

    return (4.0f / (M_PI_F * pow(h, 8.0f))) * diff * diff * diff;
}

// Spiky gradient
float2 spikyGradient(float2 rVec, float r, float h) {
    if (r <= 0.0 || r >= h) {
        return float2(0.0);
    }
    float diff = h - r;
    float coefficient = -(30.0f / (M_PI_F * pow(h, 5.0f))) * diff * diff;
    return coefficient * (rVec / r);
}

// Wall density fraction
float wallDensityFraction(float distance, float h) {
    if (distance >= h) {
        return 0.0;
    }

    float u = max(distance, 0.0f) / h;
    float t = 1.0f - u;
    return 0.5f * pow(t, 4.5f) * (1.0f + 2.2f * u + 2.2f * u * u);
}

// Density
kernel void updateDensity(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    float2 position = particles[id].position;
    float h = particles[id].smoothingRadius;
    float density = 0.0;

    for (uint j = 0; j < params.particleCount; j++) {
        float r = length(position - particles[j].position);
        density += params.mass * poly6(r, h);
    }

    density += params.restDensity * wallDensityFraction(kContainerHalfWidth  - position.x, h);
    density += params.restDensity * wallDensityFraction(kContainerHalfWidth  + position.x, h);
    density += params.restDensity * wallDensityFraction(kContainerHalfHeight - position.y, h);
    density += params.restDensity * wallDensityFraction(kContainerHalfHeight + position.y, h);

    particles[id].density = density;
}

// Smoothing radius
kernel void updateSmoothingRadius(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    float density = particles[id].density;

    float factor = 1.0 - 0.5 * (density / params.restDensity - 1.0);

    particles[id].smoothingRadius = clamp(params.smoothingRadius * factor,
                                          params.minSmoothingRadius,
                                          params.maxSmoothingRadius);
}

// Gravity
kernel void applyGravity(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    particles[id].force = float2(0.0, params.mass * params.gravity);
}

// Viscosity Laplacian
float viscosityLaplacian(float r, float h) {
    if (r >= h) {
        return 0.0;
    }
    return (40.0f / (M_PI_F * pow(h, 5.0f))) * (h - r);
}

// Pressure + Viscosity
kernel void applyPressureAndViscosity(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    float2 position = particles[id].position;
    float2 velocity = particles[id].velocity;
    float densityI = particles[id].density;
    float hI = particles[id].smoothingRadius;

    float pressureI = max(0.0f, params.pressureStiffness * (densityI - params.restDensity));
    float volumeI = params.mass / densityI;

    float2 forcePressure = float2(0.0);
    float2 forceViscous = float2(0.0);

    for (uint j = 0; j < params.particleCount; j++) {
        if (j == id) {
            continue;
        }

        float2 rVec = position - particles[j].position;
        float r = length(rVec);

        float hJ = particles[j].smoothingRadius;
        float hIJ = 0.5 * (hI + hJ);
        if (r >= hIJ) {
            continue;
        }

        float2 gradWeight = 0.5 * (spikyGradient(rVec, r, hI) + spikyGradient(rVec, r, hJ));
        float laplacianWeight = 0.5 * (viscosityLaplacian(r, hI) + viscosityLaplacian(r, hJ));

        float densityJ = particles[j].density;
        float pressureJ = max(0.0f, params.pressureStiffness * (densityJ - params.restDensity));
        float volumeJ = params.mass / densityJ;

        forcePressure -= volumeJ * (pressureI + pressureJ) * gradWeight;
        forceViscous += volumeJ * (particles[j].velocity - velocity) * laplacianWeight;
    }

    particles[id].force += volumeI * (forcePressure + params.viscosity * forceViscous);
}
