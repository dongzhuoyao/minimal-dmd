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


def compute_density_for_timestep_sampling(
    weighting_scheme='logit_normal',
    batch_size=1,
    logit_mean=0.0,
    logit_std=1.0,
    mode_scale=1.29,
    device='cpu'
):
    """
    Sample timestep values u in [0, 1] using logit-normal distribution.
    
    This is used for flow matching to sample timesteps with non-uniform distribution,
    which helps with convergence on large-scale models.
    
    Args:
        weighting_scheme: 'logit_normal' for logit-normal sampling
        batch_size: number of samples
        logit_mean: mean of the underlying normal distribution
        logit_std: standard deviation of the underlying normal distribution
        mode_scale: only used for 'mode' scheme
        device: device to place samples on
        
    Returns:
        u: [batch_size] tensor of values in [0, 1]
    """
    if weighting_scheme == 'logit_normal':
        # Sample from normal distribution
        normal_samples = torch.randn(batch_size, device=device) * logit_std + logit_mean
        # Apply sigmoid to map to (0, 1)
        u = torch.sigmoid(normal_samples)
        return u
    else:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")


class GuidanceModel(nn.Module):
    """Guidance model for DMD2 training"""
    def __init__(self, backbone, num_train_timesteps=1000, sigma_min=0.002, sigma_max=80.0, 
                 sigma_data=0.5, rho=7.0, min_step_percent=0.02, max_step_percent=0.98,
                 dynamic="vesde"):
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
        self.dynamic = dynamic.lower()  # "vesde" or "fm"
        
        if self.dynamic == "vesde":
            # Karras noise schedule for VESDE
            karras_sigmas = torch.flip(
                get_sigmas_karras(num_train_timesteps, sigma_max=sigma_max, 
                                 sigma_min=sigma_min, rho=rho),
                dims=[0]
            )
            self.register_buffer("karras_sigmas", karras_sigmas)
        elif self.dynamic == "fm":
            # For flow matching, we use time t in [0, 1]
            # Linear schedule: t goes from 0 to 1
            times = torch.linspace(0.0, 1.0, num_train_timesteps)
            self.register_buffer("times", times)
        else:
            raise ValueError(f"Unknown dynamic type: {dynamic}. Must be 'vesde' or 'fm'")
        
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
        
        if self.dynamic == "vesde":
            # VESDE-based distribution matching
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
            
        elif self.dynamic == "fm":
            # Flow matching-based distribution matching
            with torch.no_grad():
                # Difference 2: Logit-normal timestep sampling instead of uniform
                # Sample u from logit-normal distribution in [0, 1]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=batch_size,
                    logit_mean=0.0,
                    logit_std=1.0,
                    device=latents.device
                )
                # Convert u to timestep indices
                indices = (u * self.num_train_timesteps).long()
                indices = torch.clamp(indices, self.min_step, min(self.max_step, self.num_train_timesteps - 1))
                timesteps = indices
                t = u  # Use u directly as time value in [0, 1]
                
                # Sample noise (x_0 in flow matching)
                noise = torch.randn_like(latents)
                
                # Difference 1: Flow matching interpolation
                # x_t = (1 - t) * x_clean + t * x_noise
                # where x_clean is latents (clean image) and x_noise is noise
                # When t=0: x_t = x_clean, when t=1: x_t = x_noise
                t_expanded = t.view(-1, 1, 1, 1)
                x_t = (1.0 - t_expanded) * latents + t_expanded * noise
                
                # Model predicts velocity field (vector field) directly
                pred_real_velocity = self.real_unet(x_t, t, labels)
                pred_fake_velocity = self.fake_unet(x_t, t, labels)
                
                # Convert velocity field to x0 (clean image) estimation
                # From flow matching: x_t = (1-t) * x_clean + t * x_noise
                # And: v_t = x_clean - x_noise
                # Solving: x_clean = x_t + t * v_t
                pred_real_image = x_t + t_expanded * pred_real_velocity
                pred_fake_image = x_t + t_expanded * pred_fake_velocity
                
                # Now apply DMD on the x0 estimations (like SenseFlow)
                # Compute gradient direction (difference in x0 predictions)
                # p_real = latents - pred_real_image (error from real model)
                # p_fake = latents - pred_fake_image (error from fake model)
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
                "dmtrain_x_t": x_t.detach(),
                "dmtrain_pred_real_image": pred_real_image.detach(),
                "dmtrain_pred_fake_image": pred_fake_image.detach(),
                "dmtrain_grad": grad.detach(),
                "dmtrain_gradient_norm": torch.norm(grad).item(),
                "dmtrain_timesteps": timesteps.detach(),
                "dmtrain_t": t.detach(),
            }
        else:
            raise ValueError(f"Unknown dynamic type: {self.dynamic}")
        
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
        
        if self.dynamic == "vesde":
            # VESDE-based fake loss
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
            
            log_dict = {
                "faketrain_latents": latents.detach(),
                "faketrain_noisy_latents": noisy_latents.detach(),
                "faketrain_x0_pred": fake_x0_pred.detach()
            }
            
        elif self.dynamic == "fm":
            # Flow matching-based fake loss
            # Difference 2: Logit-normal timestep sampling instead of uniform
            u = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                device=latents.device
            )
            # Convert u to timestep indices
            indices = (u * self.num_train_timesteps).long()
            indices = torch.clamp(indices, 0, self.num_train_timesteps - 1)
            timesteps = indices
            t = u  # Use u directly as time value in [0, 1]
            
            # Difference 1: Flow matching interpolation
            # x_t = (1 - t) * x_clean + t * x_noise
            # where x_clean is latents (fake image) and x_noise is noise
            # When t=0: x_t = x_clean, when t=1: x_t = x_noise
            t_expanded = t.view(-1, 1, 1, 1)
            x_t = (1.0 - t_expanded) * latents + t_expanded * noise
            
            # True velocity field: v_t = x_1 - x_0 = latents - noise
            v_true = latents - noise
            
            # Model predicts velocity field (vector field) directly
            pred_velocity = self.fake_unet(x_t, t, labels)
            
            # Flow matching loss: MSE between predicted and true velocity
            loss_fake = F.mse_loss(pred_velocity, v_true, reduction="mean")
            
            log_dict = {
                "faketrain_latents": latents.detach(),
                "faketrain_x_t": x_t.detach(),
                "faketrain_pred_velocity": pred_velocity.detach(),
                "faketrain_t": t.detach(),
            }
        else:
            raise ValueError(f"Unknown dynamic type: {self.dynamic}")
        
        loss_dict = {"loss_fake_mean": loss_fake}
        
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
                 conditioning_sigma=80.0, backbone=None, dynamic="vesde"):
        super().__init__()
        
        # Create backbone model (default to SimpleUNet for backward compatibility)
        if backbone is None:
            backbone = SimpleUNet(img_channels=1, label_dim=10)
        
        self.dynamic = dynamic.lower()  # "vesde" or "fm"
        
        # Guidance model (contains real_unet and fake_unet)
        self.guidance_model = GuidanceModel(
            backbone=backbone,
            num_train_timesteps=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
            min_step_percent=min_step_percent,
            max_step_percent=max_step_percent,
            dynamic=dynamic
        )
        
        # Feedforward generator model (initialized from fake_unet)
        self.feedforward_model = copy.deepcopy(self.guidance_model.fake_unet)
        self.feedforward_model.requires_grad_(True)
        
        self.conditioning_sigma = conditioning_sigma
        self.num_train_timesteps = num_train_timesteps
        
        # For flow matching, conditioning is at t=1.0
        if self.dynamic == "fm":
            self.conditioning_time = 1.0
        else:
            self.conditioning_time = None
    
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
            # For VESDE: timestep_sigma is sigma value
            # For FM: timestep_sigma should be time t (we'll use conditioning_time)
            if self.dynamic == "fm":
                # For flow matching, use conditioning_time (t=1.0) instead of sigma
                conditioning_time = torch.ones(scaled_noisy_image.shape[0], device=scaled_noisy_image.device) * self.conditioning_time
                model_input_time = conditioning_time
            else:
                model_input_time = timestep_sigma
            
            if not compute_generator_gradient:
                with torch.no_grad():
                    generated_image = self.feedforward_model(
                        scaled_noisy_image, model_input_time, labels
                    )
            else:
                generated_image = self.feedforward_model(
                    scaled_noisy_image, model_input_time, labels
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

