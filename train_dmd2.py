"""
DMD2 Distillation Training Script for MNIST
Trains a fast feedforward model using DMD2 distillation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import os
try:
    from .config_utils import parse_args_with_optional_yaml
except ImportError:
    from config_utils import parse_args_with_optional_yaml
try:
    from .unified_model import UnifiedModel
except ImportError:
    from unified_model import UnifiedModel


def train_dmd2(args):
    """Train DMD2 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Initialize unified model
    model = UnifiedModel(
        num_train_timesteps=args.num_train_timesteps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        rho=args.rho,
        min_step_percent=args.min_step_percent,
        max_step_percent=args.max_step_percent,
        conditioning_sigma=args.conditioning_sigma
    ).to(device)
    
    # Load teacher checkpoint into real_unet
    if args.teacher_checkpoint:
        print(f"Loading teacher checkpoint from {args.teacher_checkpoint}")
        checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
        model.guidance_model.real_unet.load_state_dict(checkpoint['model_state_dict'])
        print("Teacher model loaded successfully")
    
    # Optimizers
    optimizer_generator = optim.AdamW(
        model.feedforward_model.parameters(),
        lr=args.generator_lr,
        weight_decay=0.01
    )
    
    optimizer_guidance = optim.AdamW(
        model.guidance_model.fake_unet.parameters(),
        lr=args.guidance_lr,
        weight_decay=0.01
    )
    
    # Eye matrix for one-hot encoding
    eye_matrix = torch.eye(10, device=device)
    
    # Training loop
    model.train()
    global_step = 0
    
    if args.step_number <= 0:
        raise ValueError("--step_number must be > 0 (epoch-based training has been removed).")

    data_iter = iter(train_loader)
    running_dm_sum = 0.0
    running_fake_sum = 0.0
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
            
            # Convert labels to one-hot
            labels_onehot = eye_matrix[labels]
            
            # Determine if we should compute generator gradient
            COMPUTE_GENERATOR_GRADIENT = (global_step % args.dfake_gen_update_ratio == 0)
            
            # ========== Generator Turn ==========
            # Generate scaled noise
            scaled_noise = torch.randn_like(images) * args.conditioning_sigma
            timestep_sigma = torch.ones(images.shape[0], device=device) * args.conditioning_sigma
            
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
                scaled_noise=scaled_noise,
                timestep_sigma=timestep_sigma,
                labels=gen_labels_onehot,
                real_train_dict=real_train_dict if COMPUTE_GENERATOR_GRADIENT else None,
                compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                generator_turn=True,
                guidance_turn=False
            )
            
            # Update generator if needed
            if COMPUTE_GENERATOR_GRADIENT:
                generator_loss = generator_loss_dict["loss_dm"] * args.dm_loss_weight
                
                optimizer_generator.zero_grad()
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.feedforward_model.parameters(),
                    args.max_grad_norm
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
            torch.nn.utils.clip_grad_norm_(
                model.guidance_model.fake_unet.parameters(),
                args.max_grad_norm
            )
            optimizer_guidance.step()
            optimizer_guidance.zero_grad()
            optimizer_generator.zero_grad()
            
            global_step += 1
            
            # Update progress bar
            # (these are running averages over all steps so far)
            if COMPUTE_GENERATOR_GRADIENT:
                running_dm_sum += generator_loss.item()
            running_fake_sum += guidance_loss.item()
            avg_dm = running_dm_sum / max(1, global_step // args.dfake_gen_update_ratio)
            avg_fake = running_fake_sum / max(1, global_step)
            pbar.update(1)
            pbar.set_postfix({"loss_dm": avg_dm, "loss_fake": avg_fake})
            
            # Save checkpoint periodically
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(
                    args.output_dir,
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
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "dmd2_final.pt")
    torch.save({
        'feedforward_model_state_dict': model.feedforward_model.state_dict(),
        'guidance_fake_unet_state_dict': model.guidance_model.fake_unet.state_dict(),
        'optimizer_generator_state_dict': optimizer_generator.state_dict(),
        'optimizer_guidance_state_dict': optimizer_guidance.state_dict(),
        'step': global_step,
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DMD2 model on MNIST")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional). CLI flags override YAML.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for MNIST data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/dmd2", help="Output directory for checkpoints")
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to teacher checkpoint (required; can be set via YAML)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--generator_lr", type=float, default=2e-6, help="Generator learning rate")
    parser.add_argument("--guidance_lr", type=float, default=2e-6, help="Guidance learning rate")
    parser.add_argument("--step_number", type=int, default=100_000, help="Total training steps to run (default: 100k)")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--sigma_data", type=float, default=0.5, help="Data sigma")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho parameter for Karras schedule")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="Minimum step percent for DM loss")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="Maximum step percent for DM loss")
    parser.add_argument("--conditioning_sigma", type=float, default=80.0, help="Conditioning sigma for generation")
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=10, help="Update generator every N steps")
    parser.add_argument("--dm_loss_weight", type=float, default=1.0, help="Weight for distribution matching loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")
    
    args = parse_args_with_optional_yaml(parser)
    if not args.teacher_checkpoint:
        raise ValueError("--teacher_checkpoint is required (set it via CLI or YAML config).")
    train_dmd2(args)

