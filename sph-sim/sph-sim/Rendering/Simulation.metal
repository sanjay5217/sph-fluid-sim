#include <metal_stdlib>
#include "ShaderTypes.h"
#include "SpatialGrid.h"

using namespace metal;

// Grid clear
kernel void clearGridCounts(
            device atomic_uint *cellCounts [[buffer(2)]],
            uint id [[thread_position_in_grid]]) {
    atomic_store_explicit(&cellCounts[id], 0u, memory_order_relaxed);
}

// Grid count
kernel void countGridCells(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            device atomic_uint *cellCounts [[buffer(2)]],
            uint id [[thread_position_in_grid]]) {
    uint cell = gridCellIndex(particles[id].position, params);
    atomic_fetch_add_explicit(&cellCounts[cell], 1u, memory_order_relaxed);
}

// Grid prefix sum
kernel void prefixSumGridCells(
            constant SimulationParams& params [[buffer(1)]],
            device atomic_uint *cellCounts [[buffer(2)]],
            device uint *cellStarts [[buffer(3)]],
            device atomic_uint *cellCursor [[buffer(4)]],
            uint id [[thread_position_in_grid]]) {
    if (id != 0) {
        return;
    }
    uint running = 0;
    for (uint c = 0; c < params.cellCount; c++) {
        cellStarts[c] = running;
        atomic_store_explicit(&cellCursor[c], running, memory_order_relaxed);
        running += atomic_load_explicit(&cellCounts[c], memory_order_relaxed);
    }
    cellStarts[params.cellCount] = running;
}

// Grid scatter
kernel void scatterGridParticles(
            device Particle *particles [[buffer(0)]],
            constant SimulationParams& params [[buffer(1)]],
            device atomic_uint *cellCursor [[buffer(4)]],
            device uint *sortedIndices [[buffer(5)]],
            uint id [[thread_position_in_grid]]) {
    uint cell = gridCellIndex(particles[id].position, params);
    uint slot = atomic_fetch_add_explicit(&cellCursor[cell], 1u, memory_order_relaxed);
    sortedIndices[slot] = id;
}

// Soft wall
static inline float2 softWallAcceleration(float distance, float2 normal, float2 velocity,
                                          const constant SimulationParams& params) {
    float overlap = params.wallBand - distance;
    if (overlap <= 0.0) {
        return float2(0.0);
    }

    float approach = min(dot(velocity, normal), 0.0f);
    return (params.wallStiffness * overlap - params.wallDamping * approach) * normal;
}

// Integration
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
