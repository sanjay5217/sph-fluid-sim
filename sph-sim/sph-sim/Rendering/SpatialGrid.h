#ifndef SpatialGrid_h
#define SpatialGrid_h

#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

// Grid cell
static inline uint2 gridCellCoords(float2 position, constant SimulationParams& params) {
    float2 local = position + float2(kContainerHalfWidth, kContainerHalfHeight);
    int cx = clamp(int(floor(local.x / params.cellSize)), 0, int(params.gridWidth) - 1);
    int cy = clamp(int(floor(local.y / params.cellSize)), 0, int(params.gridHeight) - 1);
    return uint2(uint(cx), uint(cy));
}

static inline uint gridCellIndex(float2 position, constant SimulationParams& params) {
    uint2 c = gridCellCoords(position, params);
    return c.y * params.gridWidth + c.x;
}

#endif
