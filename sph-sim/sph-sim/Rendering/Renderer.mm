#import "Renderer.h"
#import "ShaderTypes.h"
#include <vector>
#include <random>

static const NSUInteger kParticleCount = 10000;
static const NSUInteger kThreadsPerThreadgroup = 32;
static const float kParticleMass = 1.0f;
static const float kFixedTimestep = 0.016f;
static const float kFixedGravityConstant = -9.81f;
static const float kInitialDensity = 0.0f;
static const float kFixedSmoothingRadius = 0.2f;
static const MTLClearColor kClearColor = {0.05, 0.05, 0.08, 1.0};

@implementation Renderer {
    id <MTLDevice> _device;
    id <MTLCommandQueue> _commandQueue;
    id <MTLBuffer> _particleBuffer;
    id <MTLBuffer> _paramsBuffer;
    id <MTLRenderPipelineState> _renderPipelineState;
    id <MTLComputePipelineState> _computePipelineState;
    id <MTLComputePipelineState> _densityPipelineState;
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view {
    self = [super init];
    if (self) {
        _device = view.device;
        _commandQueue = [_device newCommandQueue];
        view.clearColor = kClearColor;

        [self buildParticleBuffer];
        [self buildPipelineWithView:view];
    }
    return self;
}


- (void)buildParticleBuffer {
    std::vector<Particle> particles(kParticleCount);

    SimulationParams params;
    params.mass = kParticleMass;
    params.dt = kFixedTimestep;
    params.gravity = kFixedGravityConstant;
    params.smoothingRadius = kFixedSmoothingRadius;
    params.particleCount = (uint)kParticleCount;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-kContainerHalfWidth, kContainerHalfWidth);


    for (auto &p : particles) {
        p.position = (vector_float2){dist(rng), dist(rng)};
        p.velocity = (vector_float2){dist(rng), dist(rng)};
        p.density = kInitialDensity;
    }

    _paramsBuffer = [_device newBufferWithBytes:&params length:sizeof(SimulationParams) options:MTLResourceStorageModeShared];
    _particleBuffer = [_device newBufferWithLength:sizeof(Particle) * kParticleCount options:MTLResourceStorageModeShared];
    memcpy(_particleBuffer.contents, particles.data(), sizeof(Particle) * kParticleCount);
}

- (void)buildPipelineWithView:(nonnull MTKView *)view {
    id <MTLLibrary> library = [_device newDefaultLibrary];
    id <MTLFunction> vertexFunction = [library newFunctionWithName:@"particleVertex"];
    id <MTLFunction> fragmentFunction = [library newFunctionWithName:@"particleFragment"];
    id <MTLFunction> computeFunction = [library newFunctionWithName:@"updateParticlePosition"];
    id <MTLFunction> densityFunction = [library newFunctionWithName:@"updateDensity"];

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
    if (!_renderPipelineState) {
        NSLog(@"Failed to create particle pipeline state: %@", error);
    }

    if (!_computePipelineState) {
        NSLog(@"Failed to create compute pipeline state: %@", error);
    }

    if (!_densityPipelineState) {
        NSLog(@"Failed to create density pipeline state: %@", error);
    }
}

- (void)drawInMTKView:(nonnull MTKView *)view {

    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"ParticleCommandBuffer";

    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor != nil) {

        // compute Encoding
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setBuffer:_particleBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:1];
        MTLSize gridSize = MTLSizeMake(kParticleCount, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);

        // Density
        [computeEncoder setComputePipelineState:_densityPipelineState];
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Particle Position
        [computeEncoder setComputePipelineState:_computePipelineState];
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];

        // render encoding
        id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"ParticleEncoder";
        [renderEncoder setRenderPipelineState:_renderPipelineState];
        [renderEncoder setVertexBuffer:_particleBuffer offset:0 atIndex:0];
        [renderEncoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1 instanceCount:kParticleCount];
        [renderEncoder endEncoding];
        [commandBuffer presentDrawable:view.currentDrawable];
    }

    [commandBuffer commit];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    // Particle positions are already in normalized device coordinates,
    // so there's no projection matrix to update yet.
}

@end
