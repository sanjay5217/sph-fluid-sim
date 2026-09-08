#include <metal_stdlib>
#include "ShaderTypes.h"
#include "SpatialGrid.h"

using namespace metal;

// poly6
static inline float poly6(float r, float h)
{
    if (r >= h) {
        return 0.0;
    }

    float h2 = h * h;
    float diff = h2 - r * r;

    return (4.0f / (M_PI_F * pow(h, 8.0f))) * diff * diff * diff;
}

// Spiky gradient
static inline float2 spikyGradient(float2 rVec, float r, float h) {
    if (r <= 0.0 || r >= h) {
        return float2(0.0);
    }
    float diff = h - r;
    float coefficient = -(30.0f / (M_PI_F * pow(h, 5.0f))) * diff * diff;
    return coefficient * (rVec / r);
}

// Viscosity Laplacian
static inline float viscosityLaplacian(float r, float h) {
    if (r >= h) {
        return 0.0;
    }
    return (40.0f / (M_PI_F * pow(h, 5.0f))) * (h - r);
}

// Wall density fraction
static inline float wallDensityFraction(float distance, float h) {
    if (distance >= h) {
        return 0.0;
    }

    float u = max(distance, 0.0f) / h;
    float t = 1.0f - u;
    return 0.5f * pow(t, 4.5f) * (1.0f + 2.2f * u + 2.2f * u * u);
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

// Density
kernel void updateDensity(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            device const uint *cellStarts [[buffer(3)]],
            device const uint *sortedIndices [[buffer(5)]],
            uint id [[thread_position_in_grid]]) {
    float2 position = particles[id].position;
    float h = particles[id].smoothingRadius;
    float density = 0.0;

    uint2 cell = gridCellCoords(position, params);
    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            int cx = int(cell.x) + ox;
            int cy = int(cell.y) + oy;
            if (cx < 0 || cy < 0 || cx >= int(params.gridWidth) || cy >= int(params.gridHeight)) {
                continue;
            }
            uint c = uint(cy) * params.gridWidth + uint(cx);
            for (uint s = cellStarts[c]; s < cellStarts[c + 1]; s++) {
                uint j = sortedIndices[s];
                float r = length(position - particles[j].position);
                density += params.mass * poly6(r, h);
            }
        }
    }

    density += params.restDensity * wallDensityFraction(kContainerHalfWidth  - position.x, h);
    density += params.restDensity * wallDensityFraction(kContainerHalfWidth  + position.x, h);
    density += params.restDensity * wallDensityFraction(kContainerHalfHeight - position.y, h);
    density += params.restDensity * wallDensityFraction(kContainerHalfHeight + position.y, h);

    particles[id].density = density;
}

// Gravity
kernel void applyGravity(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            uint id [[thread_position_in_grid]]) {
    particles[id].force = float2(0.0, params.mass * params.gravity);
}

// Pressure + Viscosity
kernel void applyPressureAndViscosity(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            device const uint *cellStarts [[buffer(3)]],
            device const uint *sortedIndices [[buffer(5)]],
            uint id [[thread_position_in_grid]]) {
    float2 position = particles[id].position;
    float2 velocity = particles[id].velocity;
    float densityI = particles[id].density;
    float hI = particles[id].smoothingRadius;

    float pressureI = max(0.0f, params.pressureStiffness * (densityI - params.restDensity));
    float volumeI = params.mass / densityI;

    float2 forcePressure = float2(0.0);
    float2 forceViscous = float2(0.0);

    uint2 cell = gridCellCoords(position, params);
    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            int cx = int(cell.x) + ox;
            int cy = int(cell.y) + oy;
            if (cx < 0 || cy < 0 || cx >= int(params.gridWidth) || cy >= int(params.gridHeight)) {
                continue;
            }
            uint c = uint(cy) * params.gridWidth + uint(cx);
            for (uint s = cellStarts[c]; s < cellStarts[c + 1]; s++) {
                uint j = sortedIndices[s];
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
        }
    }

    particles[id].force += volumeI * (forcePressure + params.viscosity * forceViscous);
}
