# DMD2 Algorithm Introduction

## Overview

**DMD2 (Distribution Matching Distillation)** is a method for distilling a slow, iterative diffusion model into a fast, single-step feedforward generator. The key insight is to match the distribution of generated images by aligning the gradient directions provided by a teacher-student guidance system, rather than directly matching the teacher's outputs.

## Core Problem

Traditional diffusion models generate images through iterative denoising (e.g., 50-1000 steps), which is slow. DMD2 aims to create a generator that produces high-quality images in a **single forward pass**, dramatically reducing inference time while maintaining image quality.

## Key Components

### 1. **Feedforward Generator** (`feedforward_model`)
- A fast UNet that generates images in one forward pass
- Takes as input: `(noise * σ_cond, σ_cond, label)`
- Outputs: a clean image directly
- This is the model we want to train and eventually use for generation

### 2. **Guidance Model** (`guidance_model`)
Contains two UNets that work together:

- **`real_unet` (Teacher)**: 
  - Frozen, pre-trained diffusion model
  - Provides "ground truth" predictions on noisy images
  - Never updated during DMD2 training

- **`fake_unet` (Student)**:
  - Trainable UNet initialized from the teacher
  - Learns to match the teacher's behavior on generated images
  - Updated every training step

### 3. **Unified Model**
Wraps both the generator and guidance model, coordinating their training.

## Algorithm Overview

DMD2 training alternates between two phases:

### **Generator Turn** (every `dfake_gen_update_ratio` steps, typically 10)

**Goal**: Train the feedforward generator to produce images that match the teacher's distribution.

**Process**:
1. **Generate image**: 
   ```
   x_gen = feedforward_model(noise * σ_cond, σ_cond, label)
   ```
   where `σ_cond = 80.0` (maximum noise level)

2. **Sample random timestep** `t` in range `[min_step, max_step]` (typically 20-980 out of 1000)
   - This avoids very noisy (early) and very clean (late) timesteps
   - Focuses on intermediate denoising steps where the signal is most informative

3. **Add noise to generated image**:
   ```
   x_noisy = x_gen + σ_t * ε
   ```
   where `ε ~ N(0, I)` and `σ_t` comes from the Karras noise schedule

4. **Get predictions from both teacher and student**:
   ```
   p_real = real_unet(x_noisy, σ_t, label)
   p_fake = fake_unet(x_noisy, σ_t, label)
   ```

5. **Compute gradient direction**:
   ```
   p_real = x_gen - p_real  # Teacher's prediction residual
   p_fake = x_gen - p_fake  # Student's prediction residual
   weight_factor = mean(|p_real|)
   grad = (p_real - p_fake) / weight_factor
   ```
   This gradient represents the direction the generator should move to better match the teacher.

6. **Distribution Matching Loss**:
   ```
   L_dm = 0.5 * MSE(x_gen, x_gen - grad)
   ```
   This encourages the generator to produce images that, when passed through the guidance model, yield predictions matching the teacher's direction.

7. **Update generator**: Backpropagate `L_dm` and update `feedforward_model` parameters.

### **Guidance Turn** (every step)

**Goal**: Train `fake_unet` to correctly denoise the generator's outputs.

**Process**:
1. **Take generated image** `x_gen` (detached, no gradient to generator)

2. **Sample random timestep** `t` uniformly from `[0, num_train_timesteps]`

3. **Add noise**:
   ```
   x_noisy = x_gen + σ_t * ε
   ```

4. **Predict clean image**:
   ```
   x̂_0 = fake_unet(x_noisy, σ_t, label)
   ```

5. **Compute loss with Karras weighting**:
   ```
   SNR = σ_t^(-2)
   weight = SNR + 1 / σ_data²
   L_fake = mean(weight * (x̂_0 - x_gen)²)
   ```
   The Karras weighting emphasizes timesteps with higher signal-to-noise ratio.

6. **Update `fake_unet`**: Backpropagate `L_fake` and update student parameters.

## Training Dynamics

The algorithm creates a **co-evolution** between the generator and guidance model:

1. **Generator** learns to produce images that match the teacher's distribution
2. **Guidance model** (`fake_unet`) learns to correctly denoise the generator's outputs
3. As the generator improves, it produces better images, which helps train `fake_unet`
4. As `fake_unet` improves, it provides better gradient signals to the generator

This creates a positive feedback loop where both models improve together.

## Key Design Choices

### 1. **Gradient Matching Instead of Direct Matching**
Instead of directly matching `fake_unet(x_noisy)` to `real_unet(x_noisy)`, DMD2 matches the **gradient direction**. This is more robust because:
- It focuses on the direction of improvement rather than absolute values
- It's normalized by the weight factor, making training more stable
- It naturally handles the distribution shift between real and generated images

### 2. **Timestep Range Restriction**
The distribution matching loss only uses timesteps in `[min_step, max_step]` (typically 2%-98% of the schedule):
- Very early timesteps (high noise) are too noisy to provide useful signal
- Very late timesteps (low noise) are too clean and don't provide enough guidance
- Intermediate timesteps contain the most informative signal

### 3. **Alternating Update Schedule**
The generator updates less frequently (`dfake_gen_update_ratio = 10`) than the guidance model:
- The guidance model needs to stay synchronized with the generator's current outputs
- More frequent guidance updates ensure `fake_unet` can provide accurate gradients
- Less frequent generator updates allow the guidance model to stabilize

### 4. **Single-Step Generation**
The feedforward model generates images using a fixed noise level (`σ_cond = 80.0`):
- This is the maximum noise level in the Karras schedule
- The model learns to denoise from this high noise level in one step
- This enables fast, single-pass generation

## Mathematical Formulation

### Distribution Matching Loss
```
L_dm = 0.5 * ||x_gen - (x_gen - grad)||²
where:
  grad = (p_real - p_fake) / weight_factor
  p_real = x_gen - real_unet(x_noisy, σ_t, label)
  p_fake = x_gen - fake_unet(x_noisy, σ_t, label)
  weight_factor = mean(|p_real|)
```

### Fake Loss (Guidance Training)
```
L_fake = mean(w_t * (fake_unet(x_noisy, σ_t, label) - x_gen)²)
where:
  w_t = σ_t^(-2) + 1 / σ_data²  (Karras weighting)
  x_noisy = x_gen + σ_t * ε
```

## Training Process Summary

```
For each training step:
  1. Sample batch (images, labels) from dataset
  
  2. GENERATOR TURN (if step % dfake_gen_update_ratio == 0):
     a. Generate: x_gen = feedforward_model(noise * σ_cond, σ_cond, random_labels)
     b. Sample timestep t in [min_step, max_step]
     c. Add noise: x_noisy = x_gen + σ_t * ε
     d. Get predictions: p_real, p_fake
     e. Compute gradient: grad = (p_real - p_fake) / weight
     f. Loss: L_dm = 0.5 * MSE(x_gen, x_gen - grad)
     g. Update feedforward_model
  
  3. GUIDANCE TURN (every step):
     a. Take generated image x_gen (detached)
     b. Sample timestep t uniformly
     c. Add noise: x_noisy = x_gen + σ_t * ε
     d. Predict: x̂_0 = fake_unet(x_noisy, σ_t, label)
     e. Loss: L_fake = mean(w_t * (x̂_0 - x_gen)²)
     f. Update fake_unet
```

## Advantages

1. **Fast Inference**: Single forward pass instead of 50-1000 iterative steps
2. **Quality Preservation**: Maintains image quality through distribution matching
3. **Stable Training**: Gradient matching and alternating updates ensure stable convergence
4. **Scalable**: Can be applied to various diffusion models and datasets

## Implementation Details

- **Noise Schedule**: Karras schedule with `σ_min = 0.002`, `σ_max = 80.0`, `ρ = 7.0`
- **Conditioning**: Class labels (one-hot encoded for MNIST)
- **Architecture**: UNet with time and label embeddings
- **Optimization**: AdamW with separate learning rates for generator and guidance model

## References

For more details on the theoretical foundations and full implementation, see the original DMD2 paper and repository.
