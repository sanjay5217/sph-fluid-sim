#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

#define kContainerWidth      2.0f
#define kContainerHeight     2.0f
#define kContainerHalfWidth  (kContainerWidth * 0.5f)
#define kContainerHalfHeight (kContainerHeight * 0.5f)

#define kMetersToNDCX (2.0f / kContainerWidth)
#define kMetersToNDCY (2.0f / kContainerHeight)

typedef struct {
    float mass;
    float dt;
    float gravity;
    float smoothingRadius;
    float minSmoothingRadius;
    float maxSmoothingRadius;
    float restDensity;
    float pressureStiffness;
    float viscosity;

    // Container walls
    float wallBand;
    float wallStiffness;
    float wallDamping;

    float pointSize;

    uint particleCount;
} SimulationParams;

typedef struct {
    vector_float2 position;
    vector_float2 velocity;
    vector_float2 force;
    float density;
    float smoothingRadius;
} Particle;

#endif
