"""
Initial training script to train a teacher diffusion model on MNIST
This creates the checkpoint that will be used for DMD2 distillation
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
    from .model import SimpleUNet, get_sigmas_karras
except ImportError:
    from model import SimpleUNet, get_sigmas_karras


@torch.no_grad()
def _sample_teacher_grid(
    model: nn.Module,
    device: torch.device,
    *,
    num_images: int,
    num_steps: int,
    conditioning_sigma: float,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_classes: int = 10,
    dynamic: str = "vesde",
) -> torch.Tensor:
    """
    Sample images from the teacher model.
    
    For VESDE: iteratively "rescaling noise" using x0 predictions.
    For FM: integrate ODE from t=0 to t=1 using velocity predictions.

    Returns a grid tensor suitable for logging (C,H,W) in [0,1].
    """
    model_was_training = model.training
    model.eval()

    B = int(num_images)
    if B <= 0:
        raise ValueError("--wandb_sample_num_images must be > 0")
    steps = int(num_steps)
    if steps <= 1:
        raise ValueError("--wandb_sample_steps must be > 1")

    # Cycle labels 0..9 to make the grid interpretable
    labels = torch.arange(B, device=device, dtype=torch.long) % int(num_classes)

    if dynamic == "vesde":
        # VESDE-based sampling
        # Sampling sigmas: go from large -> small
        sigmas = get_sigmas_karras(steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho).to(device)
        # Filter to only include sigmas <= conditioning_sigma (already in decreasing order)
        sigmas = sigmas[sigmas <= conditioning_sigma]
        if len(sigmas) == 0:
            raise ValueError(f"No sigmas <= conditioning_sigma={conditioning_sigma}. Check sigma_max >= conditioning_sigma.")
        # sigmas are already in decreasing order (sigma_max -> sigma_min), so we iterate directly

        # Start from Gaussian noise at a chosen conditioning sigma (often sigma_max)
        sigma0 = float(conditioning_sigma)
        x = torch.randn(B, 1, 28, 28, device=device) * sigma0

        # Denoise loop: move from sigma0 towards the schedule's tail.
        # We include sigma0 explicitly as the first step, then follow the Karras schedule.
        sigma_prev: float = sigma0
        for sigma_next_t in sigmas:
            sigma_next = float(sigma_next_t.item())
            sigma_prev_t = torch.full((B,), sigma_prev, device=device)
            x0_hat = model(x, sigma_prev_t, labels)
            n_hat = (x - x0_hat) / max(sigma_prev, 1e-8)
            x = x0_hat + sigma_next * n_hat
            sigma_prev = sigma_next

        # Final x0 prediction at the last sigma
        sigma_last_t = torch.full((B,), sigma_prev, device=device)
        x0 = model(x, sigma_last_t, labels)
        
    elif dynamic == "fm":
        # Flow matching-based sampling
        # Start from noise (x_0)
        x = torch.randn(B, 1, 28, 28, device=device)
        
        # Time steps from 0 to 1
        time_steps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        
        # Euler integration: x_{t+dt} = x_t + dt * v_t
        for i in range(steps):
            t = time_steps[i]
            dt = time_steps[i + 1] - time_steps[i]
            
            # Predict velocity at current time
            t_batch = torch.full((B,), t, device=device)
            v_pred = model(x, t_batch, labels)
            
            # Update x: x_{t+dt} = x_t + dt * v_t
            x = x + dt * v_pred
        
        # Final result
        x0 = x
        
    else:
        raise ValueError(f"Unknown dynamic type: {dynamic}")

    # Map from [-1,1] to [0,1]
    vis = (x0.detach().cpu() + 1.0) / 2.0
    vis = vis.clamp(0.0, 1.0)
    grid = make_grid(vis, nrow=min(8, B))

    if model_was_training:
        model.train()
    return grid


def train_teacher(cfg: DictConfig):
    """Train teacher diffusion model"""
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
    
    # Initialize model
    model = SimpleUNet(img_channels=1, label_dim=10).to(device)

    def _count_params(m: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return int(total), int(trainable)

    total_params, trainable_params = _count_params(model)
    print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")
    if wandb_run is not None:
        # Store in run summary (shows up as run-level metadata)
        wandb_run.summary["model/num_params"] = total_params
        wandb_run.summary["model/num_trainable_params"] = trainable_params
        # Also log once as scalars for convenience
        _log_wandb(
            {
                "model/num_params": total_params,
                "model/num_trainable_params": trainable_params,
            },
            step=0,
        )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    # Get dynamic type
    dynamic = cfg.dynamic.lower() if hasattr(cfg, 'dynamic') else "vesde"
    
    if dynamic == "vesde":
        # Karras noise schedule for VESDE
        sigmas = get_sigmas_karras(cfg.num_train_timesteps, 
                                   sigma_min=cfg.sigma_min,
                                   sigma_max=cfg.sigma_max,
                                   rho=cfg.rho)
        sigmas = sigmas.to(device)
        sigma_data = cfg.sigma_data
    elif dynamic == "fm":
        # Flow matching uses time t in [0, 1]
        times = torch.linspace(0.0, 1.0, cfg.num_train_timesteps, device=device)
        sigmas = None  # Not used for FM
        sigma_data = None  # Not used for FM
    else:
        raise ValueError(f"Unknown dynamic type: {dynamic}. Must be 'vesde' or 'fm'")
    
    # Training loop
    model.train()
    global_step = 0
    
    try:
        if wandb_run is not None and cfg.wandb.watch:
            import wandb  # type: ignore
            wandb.watch(model, log="all", log_freq=cfg.wandb.log_every)

        if cfg.step_number <= 0:
            raise ValueError("step_number must be > 0 (epoch-based training has been removed).")

        data_iter = iter(train_loader)
        running_loss_sum = 0.0
        pbar = tqdm(total=cfg.step_number, desc="Training", initial=global_step)

        while global_step < cfg.step_number:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)
            batch_idx = global_step % len(train_loader)

            images = images.to(device)
            labels = labels.to(device)

            # Log a quick "what does training data look like" grid once
            if wandb_run is not None and global_step == 0 and cfg.wandb.log_images:
                with torch.no_grad():
                    # images are in [-1, 1]; map to [0, 1] for visualization
                    vis = (images[: cfg.wandb.num_log_images].detach().cpu() + 1.0) / 2.0
                    grid = make_grid(vis, nrow=min(8, vis.shape[0]))
                import wandb  # type: ignore
                _log_wandb({"train/examples": wandb.Image(grid)}, step=global_step)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, cfg.num_train_timesteps,
                (images.shape[0],),
                device=device,
                dtype=torch.long
            )
            
            if dynamic == "vesde":
                timestep_sigma = sigmas[timesteps]
                
                # Add noise
                noise = torch.randn_like(images)
                noisy_images = images + timestep_sigma.reshape(-1, 1, 1, 1) * noise
                
                # Predict x0
                pred_x0 = model(noisy_images, timestep_sigma, labels)
                
                # Karras loss weighting
                snrs = timestep_sigma ** -2
                weights = snrs + 1.0 / (sigma_data ** 2)
                
                # Compute loss
                loss = torch.mean(
                    weights.reshape(-1, 1, 1, 1) * (pred_x0 - images) ** 2
                )
            elif dynamic == "fm":
                # Flow matching training
                t = times[timesteps]  # [B] time values in [0, 1]
                
                # Sample noise (x_0)
                noise = torch.randn_like(images)
                
                # Interpolate: x_t = (1 - t) * x_0 + t * x_1
                # where x_0 is noise and x_1 is the clean image
                t_expanded = t.view(-1, 1, 1, 1)
                x_t = (1 - t_expanded) * noise + t_expanded * images
                
                # True velocity: v_t = x_1 - x_0 = images - noise
                v_true = images - noise
                
                # Predict velocity
                pred_velocity = model(x_t, t, labels)
                
                # Flow matching loss: MSE between predicted and true velocity
                loss = F.mse_loss(pred_velocity, v_true, reduction="mean")
            else:
                raise ValueError(f"Unknown dynamic type: {dynamic}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            
            running_loss_sum += loss.item()
            global_step += 1
                
            # Update progress bar
            avg_loss = running_loss_sum / max(1, global_step)
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})

            if wandb_run is not None and (global_step % cfg.wandb.log_every == 0):
                lr = optimizer.param_groups[0]["lr"]
                _log_wandb(
                    {
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                    },
                    step=global_step,
                )

            # Periodically sample from the current teacher and log an image grid
            if (
                wandb_run is not None
                and cfg.wandb.log_samples
                and (global_step % cfg.wandb.sample_every == 0)
            ):
                try:
                    grid = _sample_teacher_grid(
                        model,
                        device,
                        num_images=cfg.wandb.sample_num_images,
                        num_steps=cfg.wandb.sample_steps,
                        conditioning_sigma=cfg.wandb.sample_conditioning_sigma,
                        sigma_min=cfg.sigma_min,
                        sigma_max=cfg.sigma_max,
                        rho=cfg.rho,
                        dynamic=dynamic,
                    )
                    import wandb  # type: ignore
                    _log_wandb({"samples/teacher": wandb.Image(grid)}, step=global_step)
                except Exception as e:
                    # Never crash training because sampling/logging failed
                    print(f"[wandb] sample logging failed at step {global_step}: {e}")
                
            # Save checkpoint periodically
            if global_step % cfg.save_every == 0:
                checkpoint_path = os.path.join(
                    output_dir,
                    f"teacher_checkpoint_step_{global_step}.pt"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': global_step,
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

                if wandb_run is not None and cfg.wandb.log_checkpoints:
                    import wandb  # type: ignore
                    artifact = wandb.Artifact(
                        name=f"teacher-checkpoint-step-{global_step}",
                        type="model",
                        metadata={"step": global_step},
                    )
                    artifact.add_file(checkpoint_path)
                    wandb_run.log_artifact(artifact)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(output_dir, "teacher_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': global_step,
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")


@hydra.main(version_base=None, config_path="configs", config_name="train0")
def main(cfg: DictConfig) -> None:
    train_teacher(cfg)


if __name__ == "__main__":
    main()

