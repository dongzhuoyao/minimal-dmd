# DMD2 Minimal Implementation for CIFAR-10

This folder contains a minimal, educational implementation of **DMD2 (Distribution Matching Distillation)** for CIFAR-10. DMD2 is a method to distill a slow diffusion model into a fast feedforward model.

## Overview

DMD2 consists of two main components:

1. **Feedforward Generator**: A fast model that generates images in a single forward pass
2. **Guidance Model**: Contains a frozen teacher (`real_unet`) and a trainable student (`fake_unet`) that provides training signals

The training alternates between:
- **Generator Turn**: Generate images with the feedforward model, compute distribution matching loss
- **Guidance Turn**: Train the `fake_unet` to match the `real_unet` on fake images

## Architecture

- **SimpleUNet**: A lightweight UNet architecture for 32x32 CIFAR-10 images
- **GuidanceModel**: Implements the distribution matching loss between teacher and student
- **UnifiedModel**: Wraps both generator and guidance model

## Usage

### Step 1: Train Teacher Model

First, train a teacher diffusion model that will serve as the reference:

```bash
python train_teacher.py \
    --data_dir ./data \
    --output_dir ./checkpoints/teacher \
    --batch_size 128 \
    --lr 1e-4 \
    --step_number 100000 \
    --save_every 5000
```

This will create checkpoints in `./checkpoints/teacher/`. The final checkpoint will be `teacher_final.pt`.

#### Using a YAML config

Both training scripts accept `--config` to load defaults from a YAML file (CLI flags override YAML). You’ll need PyYAML:

```bash
pip install pyyaml
```

Example:

```bash
python train_teacher.py --config config_train0.yaml
```

#### Optional: Weights & Biases logging

To enable W&B logging, install and login first:

```bash
pip install wandb
wandb login
```

Then either enable it in YAML (recommended), or via CLI.

**YAML-only (recommended)**: in your `--config` file add:

```yaml
wandb:
  enabled: true
  project: minimal-dmd
  run_name: teacher-mnist
  mode: online   # online|offline|disabled
```

**CLI**:

```bash
python train_teacher.py \
    --wandb \
    --wandb_project minimal-dmd \
    --wandb_run_name teacher-mnist \
    --wandb_dir ./log/wandb \
    --step_number 100000 \
    --wandb_log_every 50 \
    --wandb_log_images \
    --wandb_log_samples \
    --wandb_sample_every 1000 \
    --wandb_log_checkpoints
```

### Step 2: Train DMD2 Model

Use the teacher checkpoint to train the DMD2 distilled model:

```bash
python train_dmd2.py \
    --data_dir ./data \
    --output_dir ./log/checkpoints/dmd2 \
    --teacher_checkpoint ./log/checkpoints/teacher/teacher_final.pt \
    --batch_size 128 \
    --generator_lr 2e-6 \
    --guidance_lr 2e-6 \
    --step_number 100000 \
    --dfake_gen_update_ratio 10 \
    --save_every 5000
```

#### Using a YAML config

Use the dedicated DMD2 config file:

```bash
python train_dmd2.py --config config_train1.yaml
```

**Note**: Make sure `teacher_checkpoint` in `config_train1.yaml` points to a trained teacher checkpoint (default: `./log/checkpoints/teacher/teacher_final.pt`).

#### Optional: Weights & Biases logging (DMD2)

W&B logging works the same way as teacher training. You can enable it via YAML config or CLI:

**YAML** (add to your config file):
```yaml
wandb:
  enabled: true
  project: minimal-dmd
  run_name: dmd2-mnist
  log_samples: true
  sample_every: 1000
```

**CLI**:
```bash
python train_dmd2.py \
    --data_dir ./data \
    --output_dir ./log/checkpoints/dmd2 \
    --teacher_checkpoint ./log/checkpoints/teacher/teacher_final.pt \
    --wandb \
    --wandb_project minimal-dmd \
    --wandb_run_name dmd2-mnist \
    --wandb_dir ./log/wandb \
    --wandb_log_every 50 \
    --wandb_log_samples \
    --wandb_sample_every 1000 \
    --wandb_log_checkpoints
```

#### Resuming from a checkpoint

If training is interrupted, you can resume from a saved checkpoint:

```bash
python train_dmd2.py \
    --config config_train1.yaml \
    --resume_from_checkpoint ./log/checkpoints/dmd2/dmd2_checkpoint_step_50000.pt
```

Or set `resume_from_checkpoint` in `config_train1.yaml`.

This will restore:
- Model weights (feedforward_model and fake_unet)
- Optimizer states (both generator and guidance optimizers)
- Training step counter (continues from where it left off)

## Key Hyperparameters

The hyperparameters are set to match the full DMD2 implementation:

- `dfake_gen_update_ratio`: How often to update the generator (default: 10, meaning update every 10 steps). The guidance model updates every step, while the generator updates less frequently. This matches the SD implementation (ImageNet uses 5). For CIFAR-10, 10 is reasonable.
- `conditioning_sigma`: The noise level used for generation (default: 80.0). This is the maximum noise level in the Karras schedule, used for single-step generation. Matches the full implementation.
- `min_step_percent` / `max_step_percent`: Range of timesteps for distribution matching loss (default: 0.02 / 0.98). This restricts DM loss to intermediate timesteps (steps 20-980 out of 1000), avoiding very noisy and very clean timesteps. Matches the full implementation.
- `generator_lr` / `guidance_lr`: Learning rates (default: 2e-6 for both). These match the ImageNet implementation (SD uses 5e-7). For CIFAR-10, these are conservative but stable. You can experiment with 5e-6 for faster convergence, but monitor for instability.

**Note**: The learning rates are intentionally conservative to ensure stable training. If you find training too slow, you can try increasing them to 5e-6, but monitor for training instability. The other hyperparameters match the full implementation and are well-tuned.

## Files

- `model.py`: Simple UNet architecture for CIFAR-10
- `guidance.py`: Guidance model with distribution matching loss
- `unified_model.py`: Unified wrapper for generator and guidance
- `train_teacher.py`: Script to train the teacher diffusion model
- `train_dmd2.py`: Script to train the DMD2 distilled model
- `generate.py`: Script to generate images from trained model

## Generating Images

After training, you can generate images using:

```bash
python generate.py \
    --checkpoint ./log/checkpoints/dmd2/dmd2_final.pt \
    --output_dir ./generated \
    --num_samples 16
```

This will generate 16 samples for each of the 10 CIFAR-10 classes.

## Differences from Full Implementation

This minimal implementation simplifies several aspects:

1. **No GAN classifier**: The full implementation includes an optional GAN classifier for additional adversarial loss
2. **Simpler architecture**: Uses a basic UNet instead of the complex EDM architecture
3. **Single GPU**: No distributed training support
4. **No text conditioning**: CIFAR-10 uses class labels instead of text prompts

## Expected Results

After training:
- The teacher model should achieve reasonable FID scores on CIFAR-10
- The DMD2 distilled model should generate images faster (single forward pass) while maintaining quality
- Training loss should decrease over time for both generator and guidance models

## References

For the full DMD2 implementation and paper, see the main repository.

