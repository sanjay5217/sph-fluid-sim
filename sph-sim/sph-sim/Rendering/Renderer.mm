#import "Renderer.h"
#import "ShaderTypes.h"
#include <vector>
#include <random>

static const NSUInteger kThreadsPerThreadgroup = 32;
static const float kParticleMass = 1.0f;
static const float kFixedGravityConstant = -9.81f;
static const NSUInteger kMaxParticleCount = 10000;
static const NSUInteger kDefaultParticleCount = 2400;
static const float kDefaultPressureStiffness = 200.0f;
static const float kDefaultViscosity = 20.0f;
static const float kRestDensity = 1000.0f;
static const float kSmoothingRadius = 0.09f;

static const float kMinSmoothingRadiusFactor = 0.7f;
static const float kMaxSmoothingRadiusFactor = 1.5f;
static const float kFrameDuration = 1.0f / 60.0f;

static const NSUInteger kMinSubsteps = 4;
static const NSUInteger kMaxSubsteps = 40;
static const float kCFLSafetyFactor = 0.5f;

static const NSUInteger kMaxGridCells = 4096;


static const MTLClearColor kClearColor = {0.05, 0.05, 0.08, 1.0};

@implementation Renderer {
    id <MTLDevice> _device;
    id <MTLCommandQueue> _commandQueue;
    id <MTLBuffer> _particleBuffer;
    id <MTLBuffer> _paramsBuffer;
    id <MTLRenderPipelineState> _renderPipelineState;
    id <MTLComputePipelineState> _computePipelineState;
    id <MTLComputePipelineState> _densityPipelineState;
    id <MTLComputePipelineState> _smoothingPipelineState;
    id <MTLComputePipelineState> _gravityPipelineState;
    id <MTLComputePipelineState> _pressurePipelineState;
    id <MTLComputePipelineState> _clearGridPipelineState;
    id <MTLComputePipelineState> _countGridPipelineState;
    id <MTLComputePipelineState> _prefixSumPipelineState;
    id <MTLComputePipelineState> _scatterGridPipelineState;

    id <MTLBuffer> _cellCountsBuffer;
    id <MTLBuffer> _cellStartsBuffer;
    id <MTLBuffer> _cellCursorBuffer;
    id <MTLBuffer> _sortedIndicesBuffer;

    SimulationParams _params;
    NSUInteger _substepsPerFrame;
    float _drawableWidth;
    BOOL _needsRespawn;
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view {
    self = [super init];
    if (self) {
        _device = view.device;
        _commandQueue = [_device newCommandQueue];
        view.clearColor = kClearColor;

        _drawableWidth = (float)MAX(view.drawableSize.width, 1.0);
        _params.mass = kParticleMass;
        _params.gravity = kFixedGravityConstant;
        _params.smoothingRadius = kSmoothingRadius;
        _params.pressureStiffness = kDefaultPressureStiffness;
        _params.viscosity = kDefaultViscosity;
        _params.particleCount = (uint)kDefaultParticleCount;
        [self updateDerivedParameters];

        [self buildBuffers];
        [self buildPipelineWithView:view];
    }
    return self;
}

- (void)updateDerivedParameters {
    _params.restDensity = kRestDensity;

    _params.minSmoothingRadius = kSmoothingRadius * kMinSmoothingRadiusFactor;
    _params.maxSmoothingRadius = kSmoothingRadius * kMaxSmoothingRadiusFactor;

    float waveSpeed = sqrtf(fmaxf(_params.pressureStiffness, 1e-6f));
    float limit = 0.25f * _params.minSmoothingRadius / waveSpeed;
    NSUInteger needed = (NSUInteger)ceilf(kFrameDuration / fmaxf(limit * kCFLSafetyFactor, 1e-6f));
    _substepsPerFrame = MIN(kMaxSubsteps, MAX(kMinSubsteps, needed));
    _params.dt = kFrameDuration / _substepsPerFrame;

    float spacing = self.restSpacing;
    float impactSpeed = sqrtf(2.0f * fabsf(_params.gravity) * kContainerHeight);
    _params.wallBand = spacing;
    _params.wallStiffness = (impactSpeed * impactSpeed) / (2.0f * spacing * spacing);
    _params.wallDamping = 2.0f * sqrtf(_params.wallStiffness);
    _params.cellSize = _params.maxSmoothingRadius;
    _params.gridWidth = (uint)ceilf(kContainerWidth / _params.cellSize);
    _params.gridHeight = (uint)ceilf(kContainerHeight / _params.cellSize);
    _params.cellCount = _params.gridWidth * _params.gridHeight;

    [self updatePointSize];
}

- (void)updatePointSize {
    _params.pointSize = 4.0f;
}

- (void)buildBuffers {
    _paramsBuffer = [_device newBufferWithLength:sizeof(SimulationParams) options:MTLResourceStorageModeShared];
    _particleBuffer = [_device newBufferWithLength:sizeof(Particle) * kMaxParticleCount options:MTLResourceStorageModeShared];
    _cellCountsBuffer = [_device newBufferWithLength:sizeof(uint) * kMaxGridCells options:MTLResourceStorageModePrivate];
    _cellStartsBuffer = [_device newBufferWithLength:sizeof(uint) * (kMaxGridCells + 1) options:MTLResourceStorageModePrivate];
    _cellCursorBuffer = [_device newBufferWithLength:sizeof(uint) * kMaxGridCells options:MTLResourceStorageModePrivate];
    _sortedIndicesBuffer = [_device newBufferWithLength:sizeof(uint) * kMaxParticleCount options:MTLResourceStorageModePrivate];
    [self respawnParticles];
}

- (void)respawnParticles {
    std::vector<Particle> particles(_params.particleCount);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distX(-kContainerHalfWidth, kContainerHalfWidth);
    std::uniform_real_distribution<float> distY(-kContainerHalfHeight, kContainerHalfHeight);

    for (auto &p : particles) {
        p.position = (vector_float2){distX(rng), distY(rng)};
        p.velocity = (vector_float2){0.0f, 0.0f};
        p.force = (vector_float2){0.0f, 0.0f};
        p.density = _params.restDensity;
        p.smoothingRadius = _params.smoothingRadius;
    }

    memcpy(_particleBuffer.contents, particles.data(), sizeof(Particle) * particles.size());
    _needsRespawn = NO;
}

- (void)buildPipelineWithView:(nonnull MTKView *)view {
    id <MTLLibrary> library = [_device newDefaultLibrary];
    id <MTLFunction> vertexFunction = [library newFunctionWithName:@"particleVertex"];
    id <MTLFunction> fragmentFunction = [library newFunctionWithName:@"particleFragment"];
    id <MTLFunction> computeFunction = [library newFunctionWithName:@"updateParticlePosition"];
    id <MTLFunction> densityFunction = [library newFunctionWithName:@"updateDensity"];
    id <MTLFunction> smoothingFunction = [library newFunctionWithName:@"updateSmoothingRadius"];
    id <MTLFunction> gravityFunction = [library newFunctionWithName:@"applyGravity"];
    id <MTLFunction> pressureFunction = [library newFunctionWithName:@"applyPressureAndViscosity"];
    id <MTLFunction> clearGridFunction = [library newFunctionWithName:@"clearGridCounts"];
    id <MTLFunction> countGridFunction = [library newFunctionWithName:@"countGridCells"];
    id <MTLFunction> prefixSumFunction = [library newFunctionWithName:@"prefixSumGridCells"];
    id <MTLFunction> scatterGridFunction = [library newFunctionWithName:@"scatterGridParticles"];

    MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.label = @"ParticlePipeline";
    pipelineDescriptor.vertexFunction = vertexFunction;
    pipelineDescriptor.fragmentFunction = fragmentFunction;
    pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
    pipelineDescriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDescriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    NSError *error = nil;
    _renderPipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    _computePipelineState =[_device newComputePipelineStateWithFunction:computeFunction error:&error];
    _densityPipelineState = [_device newComputePipelineStateWithFunction:densityFunction error:&error];
    _smoothingPipelineState = [_device newComputePipelineStateWithFunction:smoothingFunction error:&error];
    _gravityPipelineState = [_device newComputePipelineStateWithFunction:gravityFunction error:&error];
    _pressurePipelineState = [_device newComputePipelineStateWithFunction:pressureFunction error:&error];
    _clearGridPipelineState = [_device newComputePipelineStateWithFunction:clearGridFunction error:&error];
    _countGridPipelineState = [_device newComputePipelineStateWithFunction:countGridFunction error:&error];
    _prefixSumPipelineState = [_device newComputePipelineStateWithFunction:prefixSumFunction error:&error];
    _scatterGridPipelineState = [_device newComputePipelineStateWithFunction:scatterGridFunction error:&error];
    if (!_renderPipelineState) {
        NSLog(@"Failed to create particle pipeline state: %@", error);
    }

    if (!_computePipelineState) {
        NSLog(@"Failed to create compute pipeline state: %@", error);
    }

    if (!_densityPipelineState) {
        NSLog(@"Failed to create density pipeline state: %@", error);
    }

    if (!_smoothingPipelineState) {
        NSLog(@"Failed to create smoothing radius pipeline state: %@", error);
    }

    if (!_gravityPipelineState) {
        NSLog(@"Failed to create gravity pipeline state: %@", error);
    }

    if (!_pressurePipelineState) {
        NSLog(@"Failed to create pressure pipeline state: %@", error);
    }

    if (!_clearGridPipelineState || !_countGridPipelineState ||
        !_prefixSumPipelineState || !_scatterGridPipelineState) {
        NSLog(@"Failed to create spatial grid pipeline states: %@", error);
    }
}

#pragma mark - Tunable parameters

- (NSUInteger)particleCount { return _params.particleCount; }
- (void)setParticleCount:(NSUInteger)count {
    NSUInteger clamped = MIN(MAX(count, (NSUInteger)50), kMaxParticleCount);
    if (clamped == _params.particleCount) {
        return;
    }
    _params.particleCount = (uint)clamped;
    [self updateDerivedParameters];
    _needsRespawn = YES;
}

- (float)pressureStiffness { return _params.pressureStiffness; }
- (void)setPressureStiffness:(float)stiffness {
    _params.pressureStiffness = fmaxf(stiffness, 1.0f);
    [self updateDerivedParameters];
}

- (float)viscosity { return _params.viscosity; }
- (void)setViscosity:(float)viscosity { _params.viscosity = fmaxf(viscosity, 0.0f); }

#pragma mark - Derived readouts

- (float)smoothingRadius { return _params.smoothingRadius; }
- (float)restDensity { return _params.restDensity; }
- (float)restSpacing { return sqrtf(_params.mass / _params.restDensity); }
- (float)neighbourCount {
    float h = _params.smoothingRadius;
    return M_PI * h * h * (_params.restDensity / _params.mass);
}
- (NSUInteger)substepsPerFrame { return _substepsPerFrame; }
- (float)timestep { return _params.dt; }
- (float)stableTimestepLimit {
    float c = sqrtf(fmaxf(_params.pressureStiffness, 1e-6f));
    return 0.25f * _params.smoothingRadius / c;
}

- (float)occupiedArea { return (_params.particleCount * _params.mass) / _params.restDensity; }
- (float)fillFraction { return self.occupiedArea / (kContainerWidth * kContainerHeight); }
- (NSUInteger)containerCapacity {
    return (NSUInteger)((kContainerWidth * kContainerHeight * _params.restDensity) / _params.mass);
}

- (void)resetSimulation { _needsRespawn = YES; }

#pragma mark - Frame

- (void)drawInMTKView:(nonnull MTKView *)view {
    if (_needsRespawn) {
        [self respawnParticles];
    }
    memcpy(_paramsBuffer.contents, &_params, sizeof(SimulationParams));

    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"ParticleCommandBuffer";

    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor != nil) {

        // compute Encoding
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setBuffer:_particleBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:_cellCountsBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:_cellStartsBuffer offset:0 atIndex:3];
        [computeEncoder setBuffer:_cellCursorBuffer offset:0 atIndex:4];
        [computeEncoder setBuffer:_sortedIndicesBuffer offset:0 atIndex:5];
        MTLSize gridSize = MTLSizeMake(_params.particleCount, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
        MTLSize cellGridSize = MTLSizeMake(_params.cellCount, 1, 1);
        MTLSize singleThread = MTLSizeMake(1, 1, 1);

        for (NSUInteger step = 0; step < _substepsPerFrame; step++) {

            // Smoothing radius
            [computeEncoder setComputePipelineState:_smoothingPipelineState];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

            // Neighbor search
            [computeEncoder setComputePipelineState:_clearGridPipelineState];
            [computeEncoder dispatchThreads:cellGridSize threadsPerThreadgroup:threadgroupSize];

            [computeEncoder setComputePipelineState:_countGridPipelineState];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

            [computeEncoder setComputePipelineState:_prefixSumPipelineState];
            [computeEncoder dispatchThreads:singleThread threadsPerThreadgroup:singleThread];

            [computeEncoder setComputePipelineState:_scatterGridPipelineState];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

            // Density
            [computeEncoder setComputePipelineState:_densityPipelineState];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

            // Gravity
            [computeEncoder setComputePipelineState:_gravityPipelineState];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

            // Pressure + Viscosity
            [computeEncoder setComputePipelineState:_pressurePipelineState];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

            // Integration
            [computeEncoder setComputePipelineState:_computePipelineState];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }
        [computeEncoder endEncoding];

        // render encoding
        id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"ParticleEncoder";
        [renderEncoder setRenderPipelineState:_renderPipelineState];
        [renderEncoder setVertexBuffer:_particleBuffer offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:_paramsBuffer offset:0 atIndex:1];
        [renderEncoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1 instanceCount:_params.particleCount];
        [renderEncoder endEncoding];
        [commandBuffer presentDrawable:view.currentDrawable];
    }

    [commandBuffer commit];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    _drawableWidth = (float)MAX(size.width, 1.0);
    [self updatePointSize];
}

@end
