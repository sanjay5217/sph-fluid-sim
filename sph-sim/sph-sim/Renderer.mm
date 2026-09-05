#import "Renderer.h"
#import "ShaderTypes.h"
#include <vector>
#include <random>

static const NSUInteger kParticleCount = 1000;

@implementation Renderer {
    id <MTLDevice> _device;
    id <MTLCommandQueue> _commandQueue;
    id <MTLBuffer> _particleBuffer;
    id <MTLBuffer> _paramsBuffer;
    id <MTLRenderPipelineState> _renderPipelineState;
    id <MTLComputePipelineState> _computePipelineState;
}

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view {
    self = [super init];
    if (self) {
        _device = view.device;
        _commandQueue = [_device newCommandQueue];
        view.clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0);

        [self buildParticleBuffer];
        [self buildPipelineWithView:view];
    }
    return self;
}


- (void)buildParticleBuffer {
    std::vector<Particle> particles(kParticleCount);
    
    SimulationParams params;
    params.dt = 0.016f;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);


    for (auto &p : particles) {
        p.position = (vector_float2){dist(rng), dist(rng)};
        p.velocity = (vector_float2){dist(rng), dist(rng)};
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

    MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.label = @"ParticlePipeline";
    pipelineDescriptor.vertexFunction = vertexFunction;
    pipelineDescriptor.fragmentFunction = fragmentFunction;
    pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;

    NSError *error = nil;
    _renderPipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    _computePipelineState =[_device newComputePipelineStateWithFunction:computeFunction error:&error];
    if (!_renderPipelineState) {
        NSLog(@"Failed to create particle pipeline state: %@", error);
    }
    
    if (!_computePipelineState) {
        NSLog(@"Failed to create compute pipeline state: %@", error);
    }
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    
    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"ParticleCommandBuffer";

    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor != nil) {
        
        // compute Encoding
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_computePipelineState];
        [computeEncoder setBuffer:_particleBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:_paramsBuffer offset:0 atIndex:1];
        MTLSize gridSize = MTLSizeMake(kParticleCount, 1, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
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
