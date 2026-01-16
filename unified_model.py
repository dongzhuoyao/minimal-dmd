"""
Unified Model that wraps both feedforward generator and guidance model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
try:
    from .model import SimpleUNet, get_sigmas_karras
except ImportError:
    from model import SimpleUNet, get_sigmas_karras


class GuidanceModel(nn.Module):
    """Guidance model for DMD2 training"""
    def __init__(self, backbone, num_train_timesteps=1000, sigma_min=0.002, sigma_max=80.0, 
                 sigma_data=0.5, rho=7.0, min_step_percent=0.02, max_step_percent=0.98):
        super().__init__()
        
        # Real UNet (teacher) - frozen
        self.real_unet = copy.deepcopy(backbone)
        self.real_unet.requires_grad_(False)
        
        # Fake UNet (student) - trainable
        self.fake_unet = copy.deepcopy(backbone)
        self.fake_unet.requires_grad_(True)
        
        # Training parameters
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.num_train_timesteps = num_train_timesteps
        
        # Karras noise schedule
        karras_sigmas = torch.flip(
            get_sigmas_karras(num_train_timesteps, sigma_max=sigma_max, 
                             sigma_min=sigma_min, rho=rho),
            dims=[0]
        )
        self.register_buffer("karras_sigmas", karras_sigmas)
        
        self.min_step = int(min_step_percent * num_train_timesteps)
        self.max_step = int(max_step_percent * num_train_timesteps)
    
    def compute_distribution_matching_loss(self, latents, labels):
        """
        Compute distribution matching loss between real_unet and fake_unet
        
        Args:
            latents: [B, C, H, W] clean images
            labels: [B] or [B, num_classes] class labels
            
        Returns:
            loss_dict: dictionary with loss values
            log_dict: dictionary with logging information
        """
        batch_size = latents.shape[0]
        
        # Sample random timesteps
        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step,
                min(self.max_step + 1, self.num_train_timesteps),
                [batch_size, 1, 1, 1],
                device=latents.device,
                dtype=torch.long
            )
            
            noise = torch.randn_like(latents)
            timestep_sigma = self.karras_sigmas[timesteps.squeeze()]
            
            # Add noise
            noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise
            
            # Predictions from both models
            pred_real_image = self.real_unet(noisy_latents, timestep_sigma, labels)
            pred_fake_image = self.fake_unet(noisy_latents, timestep_sigma, labels)
            
            # Compute gradient direction
            p_real = latents - pred_real_image
            p_fake = latents - pred_fake_image
            
            # Weight factor for normalization
            weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = (p_real - p_fake) / (weight_factor + 1e-8)
            grad = torch.nan_to_num(grad)
        
        # Distribution matching loss (gradient matching)
        loss = 0.5 * F.mse_loss(latents, (latents - grad).detach(), reduction="mean")
        
        loss_dict = {"loss_dm": loss}
        log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach(),
            "dmtrain_pred_real_image": pred_real_image.detach(),
            "dmtrain_pred_fake_image": pred_fake_image.detach(),
            "dmtrain_grad": grad.detach(),
            "dmtrain_gradient_norm": torch.norm(grad).item(),
            "dmtrain_timesteps": timesteps.detach(),
        }
        
        return loss_dict, log_dict
    
    def compute_loss_fake(self, latents, labels):
        """
        Compute loss for training fake_unet on fake images
        
        Args:
            latents: [B, C, H, W] fake images (detached)
            labels: [B] or [B, num_classes] class labels
            
        Returns:
            loss_dict: dictionary with loss values
            log_dict: dictionary with logging information
        """
        batch_size = latents.shape[0]
        latents = latents.detach()  # No gradient to generator
        
        noise = torch.randn_like(latents)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size, 1, 1, 1],
            device=latents.device,
            dtype=torch.long
        )
        timestep_sigma = self.karras_sigmas[timesteps.squeeze()]
        
        # Add noise
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise
        
        # Predict x0
        fake_x0_pred = self.fake_unet(noisy_latents, timestep_sigma, labels)
        
        # Karras weighting
        snrs = timestep_sigma ** -2
        weights = snrs + 1.0 / (self.sigma_data ** 2)
        
        target = latents
        
        loss_fake = torch.mean(weights.reshape(-1, 1, 1, 1) * (fake_x0_pred - target) ** 2)
        
        loss_dict = {"loss_fake_mean": loss_fake}
        log_dict = {
            "faketrain_latents": latents.detach(),
            "faketrain_noisy_latents": noisy_latents.detach(),
            "faketrain_x0_pred": fake_x0_pred.detach()
        }
        
        return loss_dict, log_dict
    
    def forward(self, generator_turn=False, guidance_turn=False,
                generator_data_dict=None, guidance_data_dict=None):
        """
        Forward pass for either generator or guidance turn
        
        Args:
            generator_turn: if True, compute distribution matching loss
            guidance_turn: if True, compute fake loss
            generator_data_dict: dict with 'image' and 'label' keys
            guidance_data_dict: dict with 'image' and 'label' keys
        """
        if generator_turn:
            assert generator_data_dict is not None
            loss_dict, log_dict = self.compute_distribution_matching_loss(
                generator_data_dict['image'],
                generator_data_dict['label']
            )
        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.compute_loss_fake(
                guidance_data_dict['image'],
                guidance_data_dict['label']
            )
        else:
            raise ValueError("Either generator_turn or guidance_turn must be True")
        
        return loss_dict, log_dict

   


class UnifiedModel(nn.Module):
    """Unified model wrapping generator and guidance"""
    def __init__(self, num_train_timesteps=1000, sigma_min=0.002, sigma_max=80.0,
                 sigma_data=0.5, rho=7.0, min_step_percent=0.02, max_step_percent=0.98,
                 conditioning_sigma=80.0, backbone=None):
        super().__init__()
        
        # Create backbone model (default to SimpleUNet for backward compatibility)
        if backbone is None:
            backbone = SimpleUNet(img_channels=1, label_dim=10)
        
        # Guidance model (contains real_unet and fake_unet)
        self.guidance_model = GuidanceModel(
            backbone=backbone,
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

