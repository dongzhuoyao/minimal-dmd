# DMD2 Architecture Overview

## Core Components

### 1. SimpleUNet (`model.py`)
A lightweight UNet architecture designed for 32x32 CIFAR-10 images:
- **Time Embedding**: Sinusoidal embeddings from noise level (sigma)
- **Label Embedding**: Embeddings for 10 CIFAR-10 classes
- **Downsampling Path**: 3 levels (64→128→256→512 channels)
- **Bottleneck**: Self-attention block
- **Upsampling Path**: 3 levels with skip connections
- **Output**: 3-channel RGB image

### 2. GuidanceModel (`guidance.py`)
Contains two UNets:
- **real_unet** (teacher): Frozen, pre-trained diffusion model
- **fake_unet** (student): Trainable, learns to match real_unet

Key methods:
- `compute_distribution_matching_loss()`: Compares predictions of real_unet vs fake_unet on noisy images
- `compute_loss_fake()`: Trains fake_unet on generated images

### 3. UnifiedModel (`unified_model.py`)
Wraps both:
- **feedforward_model**: Fast generator (single forward pass)
- **guidance_model**: Provides training signals

## Training Process

### Phase 1: Teacher Training (`train0.py`)
Standard diffusion model training:
1. Sample random timestep `t`
2. Add noise to image: `x_t = x_0 + σ_t * ε`
3. Predict clean image: `x̂_0 = UNet(x_t, σ_t, label)`
4. Compute loss: `L = w_t * ||x̂_0 - x_0||²` (Karras weighting)

### Phase 2: DMD2 Distillation (`train.py`)

**Generator Turn** (every `dfake_gen_update_ratio` steps):
1. Generate image: `x_gen = feedforward_model(noise * σ_cond, σ_cond, label)`
2. Compute distribution matching loss:
   - Sample timestep `t` in [min_step, max_step]
   - Add noise: `x_noisy = x_gen + σ_t * ε`
   - Get predictions: `p_real = real_unet(x_noisy)`, `p_fake = fake_unet(x_noisy)`
   - Compute gradient: `grad = (p_real - p_fake) / weight`
   - Loss: `L_dm = ||x_gen - (x_gen - grad)||²`
3. Update feedforward_model

**Guidance Turn** (every step):
1. Take generated image `x_gen` (detached)
2. Sample timestep `t`
3. Add noise: `x_noisy = x_gen + σ_t * ε`
4. Predict: `x̂_0 = fake_unet(x_noisy, σ_t, label)`
5. Loss: `L_fake = w_t * ||x̂_0 - x_gen||²`
6. Update fake_unet

## Key Ideas

1. **Distribution Matching**: Instead of matching the teacher's output directly, we match the gradient direction that the teacher would provide
2. **Alternating Updates**: Generator and guidance model are updated at different rates
3. **Single-Step Generation**: The feedforward model generates images in one forward pass (no iterative denoising)

## Loss Functions

### Distribution Matching Loss
```
L_dm = 0.5 * MSE(x_gen, x_gen - grad)
where grad = (pred_real - pred_fake) / weight_factor
```

This encourages the generator to produce images that, when passed through the guidance model, match the teacher's predictions.

### Fake Loss
```
L_fake = mean(w_t * (pred_fake - x_gen)²)
```

This trains the fake_unet to denoise generated images correctly.

## Hyperparameters

- `conditioning_sigma`: Noise level for generation (typically 80.0)
- `dfake_gen_update_ratio`: How often to update generator (typically 10)
- `min_step_percent` / `max_step_percent`: Timestep range for DM loss (0.02-0.98)
- `sigma_min` / `sigma_max`: Noise schedule range (0.002-80.0)

