#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

typedef struct {
    float dt;
} SimulationParams;

typedef struct {
    vector_float2 position;
    vector_float2 velocity;
} Particle;

#endif
