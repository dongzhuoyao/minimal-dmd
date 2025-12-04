"""
Initial training script to train a teacher diffusion model on CIFAR-10
This creates the checkpoint that will be used for DMD2 distillation
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
    from .model import SimpleUNet, get_sigmas_karras
except ImportError:
    from model import SimpleUNet, get_sigmas_karras


def train_teacher(args):
    """Train teacher diffusion model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    train_dataset = datasets.CIFAR10(
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
    model = SimpleUNet(img_channels=3, label_dim=10).to(device)
    
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
    
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "avg_loss": epoch_loss / (batch_idx + 1)})
            
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
                    'epoch': epoch,
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "teacher_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': global_step,
        'epoch': args.num_epochs,
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train teacher diffusion model on CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for CIFAR-10 data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/teacher", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--sigma_data", type=float, default=0.5, help="Data sigma")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho parameter for Karras schedule")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    train_teacher(args)

