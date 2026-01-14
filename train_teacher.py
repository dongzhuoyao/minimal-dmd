"""
Initial training script to train a teacher diffusion model on MNIST
This creates the checkpoint that will be used for DMD2 distillation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import argparse
import os
try:
    from .config_utils import parse_args_with_optional_yaml
except ImportError:
    from config_utils import parse_args_with_optional_yaml
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
) -> torch.Tensor:
    """
    Sample images from the teacher by iteratively "rescaling noise" using x0 predictions.

    We treat the teacher as an x0-predictor:
      x = x0 + sigma * n  =>  n_hat = (x - x0_hat) / sigma
      x_next = x0_hat + sigma_next * n_hat

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

    # Sampling sigmas: go from large -> small
    sigmas = get_sigmas_karras(steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho).to(device)
    sigmas = torch.flip(sigmas, dims=[0])

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

    # Map from [-1,1] to [0,1]
    vis = (x0.detach().cpu() + 1.0) / 2.0
    vis = vis.clamp(0.0, 1.0)
    grid = make_grid(vis, nrow=min(8, B))

    if model_was_training:
        model.train()
    return grid


def train_teacher(args):
    """Train teacher diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory (checkpoints)
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional: Weights & Biases logging
    wandb_run = None
    
    def _log_wandb(metrics: dict, step: int):
        if wandb_run is None:
            return
        import wandb  # type: ignore
        wandb.log(metrics, step=step)
    
    if args.wandb:
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "W&B logging requested (--wandb) but wandb is not installed. "
                "Install it with `pip install wandb` (and ensure you're logged in with `wandb login`)."
            ) from e

        print(
            f"[wandb] enabled: project={args.wandb_project} mode={args.wandb_mode} "
            f"run_name={args.wandb_run_name} entity={args.wandb_entity}"
        )
        os.makedirs(args.wandb_dir, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            config=vars(args),
        )
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Karras noise schedule
    sigmas = get_sigmas_karras(args.num_train_timesteps, 
                               sigma_min=args.sigma_min,
                               sigma_max=args.sigma_max,
                               rho=args.rho)
    sigmas = sigmas.to(device)
    
    sigma_data = args.sigma_data
    
    # Training loop
    model.train()
    global_step = 0
    
    try:
        if wandb_run is not None and args.wandb_watch:
            import wandb  # type: ignore
            wandb.watch(model, log="all", log_freq=args.wandb_log_every)

        if args.step_number <= 0:
            raise ValueError("--step_number must be > 0 (epoch-based training has been removed).")

        data_iter = iter(train_loader)
        running_loss_sum = 0.0
        pbar = tqdm(total=args.step_number, desc="Training", initial=global_step)

        while global_step < args.step_number:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)
            batch_idx = global_step % len(train_loader)

            images = images.to(device)
            labels = labels.to(device)

            # Log a quick "what does training data look like" grid once
            if wandb_run is not None and global_step == 0 and args.wandb_log_images:
                with torch.no_grad():
                    # images are in [-1, 1]; map to [0, 1] for visualization
                    vis = (images[: args.wandb_num_log_images].detach().cpu() + 1.0) / 2.0
                    grid = make_grid(vis, nrow=min(8, vis.shape[0]))
                import wandb  # type: ignore
                _log_wandb({"train/examples": wandb.Image(grid)}, step=global_step)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, args.num_train_timesteps,
                (images.shape[0],),
                device=device,
                dtype=torch.long
            )
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            running_loss_sum += loss.item()
            global_step += 1
                
            # Update progress bar
            avg_loss = running_loss_sum / max(1, global_step)
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})

            if wandb_run is not None and (global_step % args.wandb_log_every == 0):
                lr = optimizer.param_groups[0]["lr"]
                _log_wandb(
                    {
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/batch_idx": batch_idx,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                    },
                    step=global_step,
                )

            # Periodically sample from the current teacher and log an image grid
            if (
                wandb_run is not None
                and args.wandb_log_samples
                and (global_step % args.wandb_sample_every == 0)
            ):
                try:
                    grid = _sample_teacher_grid(
                        model,
                        device,
                        num_images=args.wandb_sample_num_images,
                        num_steps=args.wandb_sample_steps,
                        conditioning_sigma=args.wandb_sample_conditioning_sigma,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        rho=args.rho,
                    )
                    import wandb  # type: ignore
                    _log_wandb({"samples/teacher": wandb.Image(grid)}, step=global_step)
                except Exception as e:
                    # Never crash training because sampling/logging failed
                    print(f"[wandb] sample logging failed at step {global_step}: {e}")
                
            # Save checkpoint periodically
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f"teacher_checkpoint_step_{global_step}.pt"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': global_step,
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

                if wandb_run is not None and args.wandb_log_checkpoints:
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
    final_checkpoint_path = os.path.join(args.output_dir, "teacher_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': global_step,
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train teacher diffusion model on MNIST")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional). CLI flags override YAML.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument("--output_dir", type=str, default="./log/checkpoints/teacher", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--step_number", type=int, default=100_000, help="Total training steps to run (default: 100k)")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--sigma_data", type=float, default=0.5, help="Data sigma")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho parameter for Karras schedule")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")

    # Weights & Biases (optional)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="minimal-dmd", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user); optional")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name; optional")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B mode")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="W&B tags (space-separated)")
    parser.add_argument("--wandb_dir", type=str, default="./log/wandb", help="Directory to store W&B run files")
    parser.add_argument("--wandb_log_every", type=int, default=50, help="Log scalars every N steps")
    parser.add_argument("--wandb_watch", action="store_true", help="Enable wandb.watch(model)")
    parser.add_argument("--wandb_log_images", action="store_true", help="Log a small training image grid once at start")
    parser.add_argument("--wandb_num_log_images", type=int, default=32, help="Number of images to include in logged grid")
    parser.add_argument("--wandb_log_checkpoints", action="store_true", help="Log checkpoints as W&B artifacts when saved")

    # Periodic sampling (optional)
    parser.add_argument("--wandb_log_samples", action="store_true", help="Log sampled images during training")
    parser.add_argument("--wandb_sample_every", type=int, default=1000, help="Sample/log images every N steps")
    parser.add_argument("--wandb_sample_num_images", type=int, default=64, help="Number of sampled images per logged grid")
    parser.add_argument("--wandb_sample_steps", type=int, default=20, help="Number of denoising steps for teacher sampling")
    parser.add_argument(
        "--wandb_sample_conditioning_sigma",
        type=float,
        default=80.0,
        help="Initial sigma used when starting sampling from noise (often matches sigma_max)",
    )
    
    args = parse_args_with_optional_yaml(parser)
    train_teacher(args)

