//
//  Renderer.h
//  sph-sim
//
//  Created by Sanjay Ram on 2026-09-03.
//

#import <MetalKit/MetalKit.h>

// Our platform independent renderer class.   Implements the MTKViewDelegate protocol which
//   allows it to accept per-frame update and drawable resize callbacks.
@interface Renderer : NSObject <MTKViewDelegate>

-(nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view;

// Tunable inputs
@property (nonatomic) NSUInteger particleCount;
@property (nonatomic) float pressureStiffness;
@property (nonatomic) float viscosity;

// Derived readouts
@property (nonatomic, readonly) float smoothingRadius;
@property (nonatomic, readonly) float restDensity;
@property (nonatomic, readonly) float restSpacing;
@property (nonatomic, readonly) float neighbourCount;
@property (nonatomic, readonly) float occupiedArea;
@property (nonatomic, readonly) float fillFraction;
@property (nonatomic, readonly) NSUInteger containerCapacity;
@property (nonatomic, readonly) NSUInteger substepsPerFrame;
@property (nonatomic, readonly) float timestep;
@property (nonatomic, readonly) float stableTimestepLimit;

- (void)resetSimulation;

@end
