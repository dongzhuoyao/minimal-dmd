"""
Generate images using a trained DMD2 model
"""
import torch
import torch.nn as nn
from torchvision.utils import save_image
import argparse
import os
try:
    from .model import SimpleUNet
except ImportError:
    from model import SimpleUNet


def generate_images(args):
    """Generate images from trained DMD2 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = SimpleUNet(img_channels=1, label_dim=10).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['feedforward_model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # MNIST class names (digits 0-9)
    class_names = [str(i) for i in range(10)]
    
    # Get dynamic type
    dynamic = args.dynamic.lower()
    
    # Generate images
    with torch.no_grad():
        for class_idx in range(10):
            print(f"Generating images for class: {class_names[class_idx]}")
            
            # Create labels
            labels = torch.full((args.num_samples,), class_idx, device=device, dtype=torch.long)
            
            if dynamic == "vesde":
                # VESDE-based generation
                noise = torch.randn(args.num_samples, 1, 28, 28, device=device)
                scaled_noise = noise * args.conditioning_sigma
                timestep_sigma = torch.ones(args.num_samples, device=device) * args.conditioning_sigma
                # Generate images
                generated_images = model(scaled_noise, timestep_sigma, labels)
            elif dynamic == "fm":
                # Flow matching-based generation
                noise = torch.randn(args.num_samples, 1, 28, 28, device=device)
                scaled_noise = noise * args.conditioning_sigma  # Scale noise for consistency
                conditioning_time = torch.ones(args.num_samples, device=device) * 1.0
                # Generate images (model predicts at t=1.0)
                generated_images = model(scaled_noise, conditioning_time, labels)
            else:
                raise ValueError(f"Unknown dynamic type: {dynamic}")
            
            # Denormalize from [-1, 1] to [0, 1]
            generated_images = (generated_images * 0.5 + 0.5).clamp(0, 1)
            
            # Save images
            output_path = os.path.join(
                args.output_dir,
                f"{class_names[class_idx]}_samples.png"
            )
            save_image(
                generated_images,
                output_path,
                nrow=min(8, args.num_samples),
                normalize=False
            )
            print(f"Saved to {output_path}")
    
    print(f"\nAll images saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from trained DMD2 model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DMD2 checkpoint")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory for generated images")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples per class")
    parser.add_argument("--conditioning_sigma", type=float, default=80.0, help="Conditioning sigma for generation (VESDE) or noise scale (FM)")
    parser.add_argument("--dynamic", type=str, default="vesde", choices=["vesde", "fm"], help="Dynamic type: vesde or fm")
    
    args = parser.parse_args()
    generate_images(args)

