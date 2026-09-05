# sph-fluid-sim
GPU-Accelerated SPH Fluid Simulation with Apple Metal

# Particle Physics 

Before implementing SPH, I built the particle simulation incrementally to understand the physics and how it maps onto GPU computation. We start with basic **Kinematics**, which includes velocity, change in time and gravity. 

## Constant Velocity

I started with the simplest physical behavior - a particle moving at a constant velocity. Velocity has both speed and magnitude, and the position ($\mathbf{x}$) is updated as per the formula below:
\
    $$\mathbf{x}_{new} = \mathbf{x}_{old} + \mathbf{v}t$$

Letting $t = 1$, our compute kernel was simply the following:

```Objective-C++
kernel void updateParticlePosition(
            device Particle *particles [[buffer(0)]],
            uint id [[thread_position_in_grid]]) {
    Particle p = particles[id];
    p.position += p.velocity;
}
```

## Time Steps

Time steps, or $\Delta t$ is the mount of simulated time advanced during each simulation update. In other words, what amount of time passes from each update in our simulation.

```C++
static const float kFixedTimestep = 0.016f;
``` 
For now this value remains hardcoded. 

## Gravity 

Gravity is essentially the acceleration experienced by an object due to the gravitational force acting on it. The value is approximately $9.81 m/s^2$, which we denote as $g$. 

Now when we factor in acceleration, we are no longer dealing with constant velocity. Thus our equation becomes the following

\
    $$\mathbf{v}_{new} = \mathbf{v}_{old} + \mathbf{a}\Delta t$$
    $$\mathbf{x}_{new} = \mathbf{x}_{old} + \mathbf{v_new} \Delta t$$

In code:

```C++
p.velocity.y += params.gravity * params.dt;
p.position += p.velocity * params.dt;
```



