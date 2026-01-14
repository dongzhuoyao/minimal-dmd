"""
Unified Model that wraps both feedforward generator and guidance model
"""
import torch
import torch.nn as nn
import copy
try:
    from .guidance import GuidanceModel
    from .model import SimpleUNet
except ImportError:
    from guidance import GuidanceModel
    from model import SimpleUNet


class UnifiedModel(nn.Module):
    """Unified model wrapping generator and guidance"""
    def __init__(self, num_train_timesteps=1000, sigma_min=0.002, sigma_max=80.0,
                 sigma_data=0.5, rho=7.0, min_step_percent=0.02, max_step_percent=0.98,
                 conditioning_sigma=80.0):
        super().__init__()
        
        # Guidance model (contains real_unet and fake_unet)
        self.guidance_model = GuidanceModel(
            num_train_timesteps=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
            min_step_percent=min_step_percent,
            max_step_percent=max_step_percent
        )
        
        # Feedforward generator model (initialized from fake_unet)
        self.feedforward_model = copy.deepcopy(self.guidance_model.fake_unet)
        self.feedforward_model.requires_grad_(True)
        
        self.conditioning_sigma = conditioning_sigma
        self.num_train_timesteps = num_train_timesteps
    
    def forward(self, scaled_noisy_image, timestep_sigma, labels,
                real_train_dict=None,
                compute_generator_gradient=False,
                generator_turn=False,
                guidance_turn=False,
                guidance_data_dict=None):
        """
        Forward pass
        
        Args:
            scaled_noisy_image: [B, C, H, W] input noise scaled by conditioning_sigma
            timestep_sigma: [B] timestep sigma (usually conditioning_sigma)
            labels: [B] or [B, num_classes] class labels
            compute_generator_gradient: if True, compute gradients for generator
            generator_turn: if True, generator forward pass
            guidance_turn: if True, guidance forward pass
            guidance_data_dict: data dict for guidance turn
        """
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn)
        
        if generator_turn:
            # Generate image with feedforward model
            if not compute_generator_gradient:
                with torch.no_grad():
                    generated_image = self.feedforward_model(
                        scaled_noisy_image, timestep_sigma, labels
                    )
            else:
                generated_image = self.feedforward_model(
                    scaled_noisy_image, timestep_sigma, labels
                )
            
            # Compute distribution matching loss if needed
            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,
                    "label": labels,
                    "real_train_dict": real_train_dict
                }
                
                # Disable gradient for guidance model to avoid side effects
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )
                self.guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {}
            
            log_dict['generated_image'] = generated_image.detach()
            log_dict['guidance_data_dict'] = {
                "image": generated_image.detach(),
                "label": labels.detach(),
                "real_train_dict": real_train_dict
            }
        
        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict
            )
        
        return loss_dict, log_dict

