#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

// --- World-space simulation domain ---
// Position, velocity, and gravity are all in real-world units (meters,
// seconds) so that kFixedGravityConstant = -9.81 actually means what it
// says, rather than being an arbitrary number applied to an arbitrary
// coordinate range.
//
// The container is centered on the world origin: x spans
// [-kContainerHalfWidth, kContainerHalfWidth], y spans
// [-kContainerHalfHeight, kContainerHalfHeight]. Chosen as 2m x 2m
// specifically so nothing about the rendered look changes here - see
// kMetersToNDCX/Y below.
#define kContainerWidth      2.0f
#define kContainerHeight     2.0f
#define kContainerHalfWidth  (kContainerWidth * 0.5f)
#define kContainerHalfHeight (kContainerHeight * 0.5f)

// Converts a world-space (meter) position to normalized device coordinates.
// This app has always stretched the simulation to fill the window on each
// axis independently (no aspect-ratio preservation), so the conversion is a
// fixed per-axis factor rather than something recomputed on resize. With a
// 2m x 2m container this factor is exactly 1.0, so introducing real units
// here doesn't change anything about how the particles look or fill the
// window - only what the numbers mean.
#define kMetersToNDCX (2.0f / kContainerWidth)
#define kMetersToNDCY (2.0f / kContainerHeight)

typedef struct {
    float mass;
    float dt;
    float gravity;
    float smoothingRadius;
    uint particleCount;
} SimulationParams;

typedef struct {
    vector_float2 position;
    vector_float2 velocity;
    float density;
} Particle;

#endif
