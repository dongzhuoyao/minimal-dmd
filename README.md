# DMD2 Minimal Implementation for MNIST

This folder contains a minimal, educational implementation of **DMD2 (Distribution Matching Distillation)** for MNIST. DMD2 is a method to distill a slow diffusion model into a fast feedforward model.

## Overview

DMD2 consists of two main components:

1. **Feedforward Generator**: A fast model that generates images in a single forward pass
2. **Guidance Model**: Contains a frozen teacher (`real_unet`) and a trainable student (`fake_unet`) that provides training signals

The training alternates between:
- **Generator Turn**: Generate images with the feedforward model, compute distribution matching loss
- **Guidance Turn**: Train the `fake_unet` to match the `real_unet` on fake images

## Architecture

- **SimpleUNet**: A lightweight UNet architecture for 28x28 MNIST images
- **GuidanceModel**: Implements the distribution matching loss between teacher and student
- **UnifiedModel**: Wraps both generator and guidance model

## Usage

### Prerequisites

Install required dependencies:

```bash
pip install hydra-core pyyaml wandb  # wandb is optional
```

### Step 1: Train Teacher Model

The teacher training script uses **Hydra** for configuration management, just like DMD2 training. Hydra automatically creates experiment output directories (e.g., `outputs/2026-01-15/00-34-31/`) where checkpoints and logs are saved.

#### Basic Usage (Default Config)

```bash
python train0.py
```

This uses the default config at `configs/config_teacher.yaml`.

#### Using a Specific Config

```bash
python train0.py --config-name=config_teacher_train0
```

This loads `configs/config_teacher_train0.yaml` which has W&B enabled. Available configs:
- `config_teacher.yaml`: Default configuration
- `config_teacher_train0.yaml`: Example with W&B enabled

#### Overriding Config Values

You can override any config value from the command line:

```bash
# Override single values
python train0.py --config-name=config_teacher \
    batch_size=256 \
    lr=2e-4 \
    step_number=200000

# Override nested values (wandb)
python train0.py --config-name=config_teacher \
    wandb.enabled=true \
    wandb.project=my-project \
    wandb.run_name=teacher-experiment
```

#### Output Directory Structure

Hydra automatically creates output directories for each run:

```
outputs/
└── 2026-01-15/
    └── 00-34-31/
        ├── checkpoints/          # Model checkpoints
        │   ├── teacher_checkpoint_step_5000.pt
        │   └── teacher_final.pt
        ├── wandb/                # W&B logs (if enabled)
        └── .hydra/                # Hydra config backups
            └── config.yaml
```

#### Optional: Weights & Biases logging

To enable W&B logging, install and login first:

```bash
pip install wandb
wandb login
```

Enable W&B in your config file:

```yaml
wandb:
  enabled: true
  project: minimal-dmd
  run_name: teacher-mnist
  mode: online  # online|offline|disabled
  log_samples: true
  sample_every: 1000
```

Or override from command line:

```bash
python train0.py --config-name=config_teacher \
    wandb.enabled=true \
    wandb.run_name=teacher-experiment \
    wandb.log_samples=true
```

### Step 2: Train DMD2 Model

The DMD2 training script uses **Hydra** for configuration management. Hydra automatically creates experiment output directories (e.g., `outputs/2026-01-15/00-34-31/`) where checkpoints and logs are saved.

#### Basic Usage (Default Config)

```bash
python train.py
```

This uses the default config at `configs/config.yaml`. **Note**: You must set `teacher_checkpoint` in the config file or override it via command line.

#### Using a Specific Config

```bash
python train.py --config-name=config_train1
```

This loads `configs/config_train1.yaml`. Available configs:
- `config.yaml`: Default configuration
- `config_train0.yaml`: Example with W&B enabled
- `config_train1.yaml`: Example with specific teacher checkpoint path

#### Overriding Config Values

You can override any config value from the command line:

```bash
# Override single values
python train.py --config-name=config_train1 \
    batch_size=256 \
    generator_lr=1e-5 \
    teacher_checkpoint=./log/checkpoints/teacher/teacher_final.pt

# Override nested values (wandb)
python train.py --config-name=config_train1 \
    wandb.enabled=true \
    wandb.project=my-project \
    wandb.run_name=experiment1
```

#### Output Directory Structure

Hydra automatically creates output directories for each run:

```
outputs/
└── 2026-01-15/
    └── 00-34-31/
        ├── checkpoints/          # Model checkpoints
        │   ├── dmd2_checkpoint_step_5000.pt
        │   └── dmd2_final.pt
        ├── wandb/                # W&B logs (if enabled)
        └── .hydra/                # Hydra config backups
            └── config.yaml
```

#### Optional: Weights & Biases logging (DMD2)

Enable W&B in your config file:

```yaml
wandb:
  enabled: true
  project: minimal-dmd
  run_name: dmd2-mnist
  mode: online  # online|offline|disabled
  log_samples: true
  sample_every: 1000
```

Or override from command line:

```bash
python train.py --config-name=config_train1 \
    wandb.enabled=true \
    wandb.run_name=dmd2-experiment \
    wandb.log_samples=true
```

#### Resuming from a checkpoint

If training is interrupted, you can resume from a saved checkpoint:

```bash
python train.py --config-name=config_train1 \
    resume_from_checkpoint=./outputs/2026-01-15/00-34-31/checkpoints/dmd2_checkpoint_step_50000.pt
```

Or set `resume_from_checkpoint` in your config file.

This will restore:
- Model weights (feedforward_model and fake_unet)
- Optimizer states (both generator and guidance optimizers)
- Training step counter (continues from where it left off)

#### Config File Structure

Config files are located in `configs/` directory. Example structure:

```yaml
# @package _global_

# Data and I/O (paths relative to original working directory)
data_dir: ./data
teacher_checkpoint: ./log/checkpoints/teacher/teacher_final.pt

# Output directories (relative to Hydra experiment output directory)
output_dir: checkpoints  # Creates outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/
wandb:
  dir: wandb  # Creates outputs/YYYY-MM-DD/HH-MM-SS/wandb/

# Training hyperparameters
batch_size: 128
generator_lr: 2.0e-6
guidance_lr: 2.0e-6
step_number: 100000
# ... etc
```

See `configs/README.md` for more details on Hydra usage.

## Key Hyperparameters

The hyperparameters are set to match the full DMD2 implementation:

- `dfake_gen_update_ratio`: How often to update the generator (default: 10, meaning update every 10 steps). The guidance model updates every step, while the generator updates less frequently. This matches the SD implementation (ImageNet uses 5). For MNIST, 10 is reasonable.
- `conditioning_sigma`: The noise level used for generation (default: 80.0). This is the maximum noise level in the Karras schedule, used for single-step generation. Matches the full implementation.
- `min_step_percent` / `max_step_percent`: Range of timesteps for distribution matching loss (default: 0.02 / 0.98). This restricts DM loss to intermediate timesteps (steps 20-980 out of 1000), avoiding very noisy and very clean timesteps. Matches the full implementation.
- `generator_lr` / `guidance_lr`: Learning rates (default: 2e-6 for both). These match the ImageNet implementation (SD uses 5e-7). For MNIST, these are conservative but stable. You can experiment with 5e-6 for faster convergence, but monitor for instability.

**Note**: The learning rates are intentionally conservative to ensure stable training. If you find training too slow, you can try increasing them to 5e-6, but monitor for training instability. The other hyperparameters match the full implementation and are well-tuned.

## Files

- `model.py`: Simple UNet architecture for MNIST
- `guidance.py`: Guidance model with distribution matching loss
- `unified_model.py`: Unified wrapper for generator and guidance
- `train0.py`: Script to train the teacher diffusion model (uses Hydra)
- `train.py`: Script to train the DMD2 distilled model (uses Hydra)
- `generate.py`: Script to generate images from trained model
- `configs/`: Hydra configuration files
  - `config_teacher.yaml`: Default teacher training configuration
  - `config_teacher_train0.yaml`: Teacher training with W&B enabled
  - `config.yaml`: Default DMD2 training configuration
  - `config_train0.yaml`: DMD2 training with W&B enabled
  - `config_train1.yaml`: DMD2 training with specific settings
  - `README.md`: Detailed Hydra usage guide

## Generating Images

After training, you can generate images using:

```bash
python generate.py \
    --checkpoint ./outputs/2026-01-15/00-34-31/checkpoints/dmd2_final.pt \
    --output_dir ./generated \
    --num_samples 16
```

**Note**: Checkpoints are now saved in Hydra output directories (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`). Replace the path with your actual experiment output directory.

This will generate 16 samples for each of the 10 classes (MNIST has 10 digit classes).

## Differences from Full Implementation

This minimal implementation simplifies several aspects:

1. **No GAN classifier**: The full implementation includes an optional GAN classifier for additional adversarial loss
2. **Simpler architecture**: Uses a basic UNet instead of the complex EDM architecture
3. **Single GPU**: No distributed training support
4. **No text conditioning**: MNIST uses class labels instead of text prompts

## Expected Results

After training:
- The teacher model should achieve reasonable FID scores on MNIST
- The DMD2 distilled model should generate images faster (single forward pass) while maintaining quality
- Training loss should decrease over time for both generator and guidance models
- All outputs (checkpoints, logs) are organized in Hydra experiment directories (`outputs/YYYY-MM-DD/HH-MM-SS/`)

## References

For the full DMD2 implementation and paper, see the main repository.

