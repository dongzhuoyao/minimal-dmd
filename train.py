"""
DMD2 Distillation Training Script for MNIST
Trains a fast feedforward model using DMD2 distillation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
try:
    from .unified_model import UnifiedModel
    from .model import get_sigmas_karras
except ImportError:
    from unified_model import UnifiedModel
    from model import get_sigmas_karras


@torch.no_grad()
def _sample_dmd2_grid(
    feedforward_model: nn.Module,
    device: torch.device,
    *,
    num_images: int,
    conditioning_sigma: float,
    num_classes: int = 10,
    dynamic: str = "vesde",
) -> torch.Tensor:
    """
    Sample images from the distilled feedforward model (single forward pass).

    Returns a grid tensor (C,H,W) in [0,1] suitable for W&B.
    """
    model_was_training = feedforward_model.training
    feedforward_model.eval()

    B = int(num_images)
    if B <= 0:
        raise ValueError("--wandb_sample_num_images must be > 0")

    labels = torch.arange(B, device=device, dtype=torch.long) % int(num_classes)
    
    if dynamic == "vesde":
        sigma = float(conditioning_sigma)
        scaled_noise = torch.randn(B, 1, 28, 28, device=device) * sigma
        timestep_sigma = torch.ones(B, device=device) * sigma
        x0 = feedforward_model(scaled_noise, timestep_sigma, labels)
    elif dynamic == "fm":
        # For flow matching, start from noise and use t=1.0
        scaled_noise = torch.randn(B, 1, 28, 28, device=device) * conditioning_sigma
        conditioning_time = torch.ones(B, device=device) * 1.0
        x0 = feedforward_model(scaled_noise, conditioning_time, labels)
    else:
        raise ValueError(f"Unknown dynamic type: {dynamic}")
    
    vis = (x0.detach().cpu() + 1.0) / 2.0
    vis = vis.clamp(0.0, 1.0)
    grid = make_grid(vis, nrow=min(8, B))

    if model_was_training:
        feedforward_model.train()
    return grid


@torch.no_grad()
def _sample_teacher_grid(
    teacher_model: nn.Module,
    device: torch.device,
    *,
    num_images: int,
    conditioning_sigma: float,
    num_classes: int = 10,
    dynamic: str = "vesde",
) -> torch.Tensor:
    """
    Sample images from the teacher model using single forward pass (same as DMD for fair comparison).
    
    This samples from the initial teacher checkpoint to compare with DMD samples.
    Both use single-step sampling for fair comparison.

    Returns a grid tensor (C,H,W) in [0,1] suitable for W&B.
    """
    model_was_training = teacher_model.training
    teacher_model.eval()

    B = int(num_images)
    if B <= 0:
        raise ValueError("--wandb_sample_num_images must be > 0")

    # Cycle labels 0..9 to make the grid interpretable (same as DMD)
    labels = torch.arange(B, device=device, dtype=torch.long) % int(num_classes)
    
    if dynamic == "vesde":
        sigma = float(conditioning_sigma)
        # Single-step sampling: same as DMD
        scaled_noise = torch.randn(B, 1, 28, 28, device=device) * sigma
        timestep_sigma = torch.ones(B, device=device) * sigma
        # Single forward pass
        x0 = teacher_model(scaled_noise, timestep_sigma, labels)
    elif dynamic == "fm":
        # For flow matching, start from noise and use t=1.0
        scaled_noise = torch.randn(B, 1, 28, 28, device=device)
        conditioning_time = torch.ones(B, device=device) * 1.0
        x0 = teacher_model(scaled_noise, conditioning_time, labels)
    else:
        raise ValueError(f"Unknown dynamic type: {dynamic}")
    
    # Map from [-1,1] to [0,1]
    vis = (x0.detach().cpu() + 1.0) / 2.0
    vis = vis.clamp(0.0, 1.0)
    grid = make_grid(vis, nrow=min(8, B))

    if model_was_training:
        teacher_model.train()
    return grid


def train(cfg: DictConfig):
    """Train DMD2 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get Hydra output directory - Hydra changes working directory to the experiment output directory
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    print(f"Hydra output directory: {hydra_output_dir}")
    
    # Create checkpoints directory under Hydra output directory
    output_dir = os.path.join(hydra_output_dir, cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Checkpoints directory: {output_dir}")

    # Optional: Weights & Biases logging
    wandb_run = None
    
    def _log_wandb(metrics: dict, step: int):
        if wandb_run is None:
            return
        import wandb  # type: ignore
        wandb.log(metrics, step=step)
    
    if cfg.wandb.enabled:
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "W&B logging requested (--wandb) but wandb is not installed. "
                "Install it with `pip install wandb` (and ensure you're logged in with `wandb login`)."
            ) from e

        # Get config name and dynamic value to construct run name
        config_name = hydra_cfg.job.config_name
        dynamic = cfg.dynamic.lower() if hasattr(cfg, 'dynamic') else "vesde"
        wandb_run_name = f"{config_name}-{dynamic}"
        
        print(
            f"[wandb] enabled: project={cfg.wandb.project} mode={cfg.wandb.mode} "
            f"run_name={wandb_run_name} entity={cfg.wandb.entity}"
        )
        # Create wandb directory under Hydra output directory
        wandb_dir = os.path.join(hydra_output_dir, cfg.wandb.dir)
        os.makedirs(wandb_dir, exist_ok=True)
        print(f"W&B directory: {wandb_dir}")
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=wandb_run_name,
            tags=cfg.wandb.tags if cfg.wandb.tags is not None else None,
            mode=cfg.wandb.mode,
            dir=wandb_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # data_dir is relative to original working directory
    original_cwd = get_original_cwd()
    data_dir = os.path.join(original_cwd, cfg.data_dir) if not os.path.isabs(cfg.data_dir) else cfg.data_dir
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get dynamic type
    dynamic = cfg.dynamic.lower() if hasattr(cfg, 'dynamic') else "vesde"
    
    # Initialize unified model
    model = UnifiedModel(
        num_train_timesteps=cfg.num_train_timesteps,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        sigma_data=cfg.sigma_data,
        rho=cfg.rho,
        min_step_percent=cfg.min_step_percent,
        max_step_percent=cfg.max_step_percent,
        conditioning_sigma=cfg.conditioning_sigma,
        dynamic=dynamic
    ).to(device)

    def _count_params(m: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return int(total), int(trainable)

    # Log param counts once
    total_params, trainable_params = _count_params(model)
    gen_total, gen_trainable = _count_params(model.feedforward_model)
    fake_total, fake_trainable = _count_params(model.guidance_model.fake_unet)
    print(
        "Model parameters: "
        f"unified(total={total_params:,} trainable={trainable_params:,}) "
        f"generator(total={gen_total:,} trainable={gen_trainable:,}) "
        f"fake_unet(total={fake_total:,} trainable={fake_trainable:,})"
    )
    if wandb_run is not None:
        wandb_run.summary["model/num_params"] = total_params
        wandb_run.summary["model/num_trainable_params"] = trainable_params
        wandb_run.summary["model/generator_num_params"] = gen_total
        wandb_run.summary["model/generator_num_trainable_params"] = gen_trainable
        wandb_run.summary["model/fake_unet_num_params"] = fake_total
        wandb_run.summary["model/fake_unet_num_trainable_params"] = fake_trainable
        _log_wandb(
            {
                "model/num_params": total_params,
                "model/num_trainable_params": trainable_params,
                "model/generator_num_params": gen_total,
                "model/generator_num_trainable_params": gen_trainable,
                "model/fake_unet_num_params": fake_total,
                "model/fake_unet_num_trainable_params": fake_trainable,
            },
            step=0,
        )
    
    # Load teacher checkpoint into real_unet (required)
    if not cfg.teacher_checkpoint:
        raise ValueError("teacher_checkpoint is required (set it in config file).")
    
    # teacher_checkpoint is relative to original working directory
    teacher_checkpoint_path = os.path.join(original_cwd, cfg.teacher_checkpoint) if not os.path.isabs(cfg.teacher_checkpoint) else cfg.teacher_checkpoint
    
    if not os.path.exists(teacher_checkpoint_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_checkpoint_path}\n"
            f"Please train a teacher model first using train0.py"
        )
    
    print(f"Loading teacher checkpoint from {teacher_checkpoint_path}")
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    
    # Load teacher model weights into real_unet (frozen teacher)
    if 'model_state_dict' in teacher_checkpoint:
        model.guidance_model.real_unet.load_state_dict(teacher_checkpoint['model_state_dict'])
        print("Teacher model (real_unet) loaded successfully")
    else:
        raise ValueError(
            f"Teacher checkpoint missing 'model_state_dict' key. "
            f"Found keys: {list(teacher_checkpoint.keys())}"
        )
    
    # Ensure real_unet is frozen
    model.guidance_model.real_unet.requires_grad_(False)
    model.guidance_model.real_unet.eval()
    print("Teacher model (real_unet) is frozen")
    
    # Optimizers
    optimizer_generator = optim.AdamW(
        model.feedforward_model.parameters(),
        lr=cfg.generator_lr,
        weight_decay=0.01
    )
    
    optimizer_guidance = optim.AdamW(
        model.guidance_model.fake_unet.parameters(),
        lr=cfg.guidance_lr,
        weight_decay=0.01
    )
    
    # Eye matrix for one-hot encoding
    eye_matrix = torch.eye(10, device=device)
    
    # Training loop
    model.train()
    global_step = 0
    
    # Checkpoint resuming support
    if cfg.resume_from_checkpoint:
        # resume_from_checkpoint can be relative to original working directory or absolute path
        resume_checkpoint_path = os.path.join(original_cwd, cfg.resume_from_checkpoint) if not os.path.isabs(cfg.resume_from_checkpoint) else cfg.resume_from_checkpoint
        
        if not os.path.exists(resume_checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        
        # Load model states
        if 'feedforward_model_state_dict' in resume_checkpoint:
            model.feedforward_model.load_state_dict(resume_checkpoint['feedforward_model_state_dict'])
            print("  ✓ Feedforward model loaded")
        
        if 'guidance_fake_unet_state_dict' in resume_checkpoint:
            model.guidance_model.fake_unet.load_state_dict(resume_checkpoint['guidance_fake_unet_state_dict'])
            print("  ✓ Guidance fake_unet loaded")
        
        # Load optimizer states
        if 'optimizer_generator_state_dict' in resume_checkpoint:
            optimizer_generator.load_state_dict(resume_checkpoint['optimizer_generator_state_dict'])
            print("  ✓ Generator optimizer loaded")
        
        if 'optimizer_guidance_state_dict' in resume_checkpoint:
            optimizer_guidance.load_state_dict(resume_checkpoint['optimizer_guidance_state_dict'])
            print("  ✓ Guidance optimizer loaded")
        
        # Resume from step
        if 'step' in resume_checkpoint:
            global_step = resume_checkpoint['step']
            print(f"  ✓ Resuming from step {global_step}")
        
        print("Resume complete!")
    
    if cfg.step_number <= 0:
        raise ValueError("step_number must be > 0 (epoch-based training has been removed).")

    data_iter = iter(train_loader)
    running_dm_sum = 0.0
    running_fake_sum = 0.0
    pbar = tqdm(total=cfg.step_number, desc="Training", initial=global_step)
    try:
        if wandb_run is not None and cfg.wandb.watch:
            import wandb  # type: ignore
            wandb.watch(model, log="all", log_freq=cfg.wandb.log_every)

        # Log a quick "what does training data look like" grid once
        if wandb_run is not None and cfg.wandb.log_images:
            try:
                images0, _labels0 = next(iter(train_loader))
                # images are in [-1, 1]; map to [0, 1] for visualization
                vis = (images0[: cfg.wandb.num_log_images].detach().cpu() + 1.0) / 2.0
                grid = make_grid(vis, nrow=min(8, vis.shape[0]))
                import wandb  # type: ignore
                _log_wandb({"train/examples": wandb.Image(grid)}, step=0)
            except Exception as e:
                print(f"[wandb] failed to log train/examples: {e}")

        while global_step < cfg.step_number:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)
            
            
            images = images.to(device)
            labels = labels.to(device)

            # Convert labels to one-hot
            labels_onehot = eye_matrix[labels]

            # Determine if we should compute generator gradient
            COMPUTE_GENERATOR_GRADIENT = (global_step % cfg.dfake_gen_update_ratio == 0)

            # ========== Generator Turn ==========
            # Generate scaled noise
            scaled_noise = torch.randn_like(images) * cfg.conditioning_sigma
            timestep_sigma = torch.ones(images.shape[0], device=device) * cfg.conditioning_sigma

            # Random labels for generation
            gen_labels = torch.randint(0, 10, (images.shape[0],), device=device)
            gen_labels_onehot = eye_matrix[gen_labels]

            # Real training dict (for optional GAN loss)
            real_train_dict = {
                "real_image": images,
                "real_label": labels_onehot
            }

            # Forward pass through generator
            generator_loss_dict, generator_log_dict = model(
                scaled_noisy_image=scaled_noise,
                timestep_sigma=timestep_sigma,
                labels=gen_labels_onehot,
                real_train_dict=real_train_dict if COMPUTE_GENERATOR_GRADIENT else None,
                compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                generator_turn=True,
                guidance_turn=False
            )

            gen_grad_norm = None
            generator_loss = None

            # Update generator if needed
            if COMPUTE_GENERATOR_GRADIENT:
                generator_loss = generator_loss_dict["loss_dm"] * cfg.dm_loss_weight

                optimizer_generator.zero_grad()
                generator_loss.backward()
                gen_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.feedforward_model.parameters(),
                    cfg.max_grad_norm
                )
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_guidance.zero_grad()

            # ========== Guidance Turn ==========
            # Update guidance model (fake_unet)
            guidance_loss_dict, guidance_log_dict = model(
                scaled_noisy_image=None,  # Not used in guidance turn
                timestep_sigma=None,  # Not used in guidance turn
                labels=None,  # Not used in guidance turn
                compute_generator_gradient=False,
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=generator_log_dict['guidance_data_dict']
            )

            guidance_loss = guidance_loss_dict["loss_fake_mean"]

            optimizer_guidance.zero_grad()
            guidance_loss.backward()
            fake_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.guidance_model.fake_unet.parameters(),
                cfg.max_grad_norm
            )
            optimizer_guidance.step()
            optimizer_guidance.zero_grad()
            optimizer_generator.zero_grad()

            global_step += 1

            # Update progress bar
            # (these are running averages over all steps so far)
            if COMPUTE_GENERATOR_GRADIENT and generator_loss is not None:
                running_dm_sum += float(generator_loss.item())
            running_fake_sum += float(guidance_loss.item())
            avg_dm = running_dm_sum / max(1, global_step // cfg.dfake_gen_update_ratio)
            avg_fake = running_fake_sum / max(1, global_step)
            pbar.update(1)
            pbar.set_postfix({"loss_dm": avg_dm, "loss_fake": avg_fake})

            # W&B scalar logging
            if wandb_run is not None and (global_step % cfg.wandb.log_every == 0):
                metrics = {
                    "train/loss_fake": float(guidance_loss.item()),
                    "train/avg_loss_fake": float(avg_fake),
                    "train/avg_loss_dm": float(avg_dm),  # Always log running average
                    "train/guidance_lr": float(optimizer_guidance.param_groups[0]["lr"]),
                    "train/guidance_grad_norm": float(fake_grad_norm),
                }
                # Log current loss_dm and generator metrics only when computed
                if COMPUTE_GENERATOR_GRADIENT and generator_loss is not None:
                    metrics.update(
                        {
                            "train/loss_dm": float(generator_loss.item()),
                            "train/generator_lr": float(optimizer_generator.param_groups[0]["lr"]),
                            "train/generator_grad_norm": float(gen_grad_norm) if gen_grad_norm is not None else 0.0,
                        }
                    )
                _log_wandb(metrics, step=global_step)

            # Periodically sample from both DMD and teacher for comparison
            if (
                wandb_run is not None
                and cfg.wandb.log_samples
                and (global_step % cfg.wandb.sample_every == 0)
            ):
                try:
                    import wandb  # type: ignore
                    
                    # Sample from DMD model (single forward pass)
                    dmd_grid = _sample_dmd2_grid(
                        model.feedforward_model,
                        device,
                        num_images=cfg.wandb.sample_num_images,
                        conditioning_sigma=cfg.conditioning_sigma,
                        dynamic=dynamic,
                    )
                    _log_wandb({"samples/dmd2": wandb.Image(dmd_grid)}, step=global_step)
                    
                    # Sample from teacher model (single-step, same as DMD for fair comparison)
                    teacher_grid = _sample_teacher_grid(
                        model.guidance_model.real_unet,
                        device,
                        num_images=cfg.wandb.sample_num_images,
                        conditioning_sigma=cfg.conditioning_sigma,
                        dynamic=dynamic,
                    )
                    _log_wandb({"samples/teacher": wandb.Image(teacher_grid)}, step=global_step)
                    
                    # Create side-by-side comparison (teacher left, DMD right)
                    # Ensure both grids have the same height for proper alignment
                    if teacher_grid.shape[1] != dmd_grid.shape[1]:
                        # Resize to match height (shouldn't happen with same num_images, but just in case)
                        target_h = min(teacher_grid.shape[1], dmd_grid.shape[1])
                        if teacher_grid.shape[1] != target_h:
                            teacher_grid = F.interpolate(
                                teacher_grid.unsqueeze(0), 
                                size=(target_h, teacher_grid.shape[2]), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                        if dmd_grid.shape[1] != target_h:
                            dmd_grid = F.interpolate(
                                dmd_grid.unsqueeze(0), 
                                size=(target_h, dmd_grid.shape[2]), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                    # Concatenate horizontally (side-by-side): teacher left, DMD right
                    comparison_grid = torch.cat([teacher_grid, dmd_grid], dim=2)  # dim=2 is width
                    _log_wandb({"samples/comparison": wandb.Image(comparison_grid)}, step=global_step)
                        
                except Exception as e:
                    print(f"[wandb] sample logging failed at step {global_step}: {e}")

            # Save checkpoint periodically
            if global_step % cfg.save_every == 0:
                checkpoint_path = os.path.join(
                    output_dir,
                    f"dmd2_checkpoint_step_{global_step}.pt"
                )
                torch.save({
                    'feedforward_model_state_dict': model.feedforward_model.state_dict(),
                    'guidance_fake_unet_state_dict': model.guidance_model.fake_unet.state_dict(),
                    'optimizer_generator_state_dict': optimizer_generator.state_dict(),
                    'optimizer_guidance_state_dict': optimizer_guidance.state_dict(),
                    'step': global_step,
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

                if wandb_run is not None and cfg.wandb.log_checkpoints:
                    import wandb  # type: ignore
                    artifact = wandb.Artifact(
                        name=f"dmd2-checkpoint-step-{global_step}",
                        type="model",
                        metadata={"step": global_step},
                    )
                    artifact.add_file(checkpoint_path)
                    wandb_run.log_artifact(artifact)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(output_dir, "dmd2_final.pt")
    torch.save({
        'feedforward_model_state_dict': model.feedforward_model.state_dict(),
        'guidance_fake_unet_state_dict': model.guidance_model.fake_unet.state_dict(),
        'optimizer_generator_state_dict': optimizer_generator.state_dict(),
        'optimizer_guidance_state_dict': optimizer_guidance.state_dict(),
        'step': global_step,
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")


@hydra.main(version_base=None, config_path="configs", config_name="train1")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
