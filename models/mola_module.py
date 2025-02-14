import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional



#Based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
def _step_with_mpgd(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        edit_scale,
        length,
        max_step,
        vae_model,
        feats2joints=None,
        control= None, 
        repeat: int = 1, 
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        for travel in range(repeat):
            # 1. get previous step value (=t-1)
            prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

            # 2. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

            beta_prod_t = 1 - alpha_prod_t
            alpha_s = alpha_prod_t / alpha_prod_t_prev

            # 3. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            if self.config.prediction_type == "epsilon":
                pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                pred_epsilon = model_output
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
                pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            elif self.config.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
                pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction`"
                )

            # 4. Clip or threshold "predicted x_0"
            if self.config.thresholding:
                pred_original_sample = self._threshold_sample(pred_original_sample)
            elif self.config.clip_sample:
                pred_original_sample = pred_original_sample.clamp(
                    -self.config.clip_sample_range, self.config.clip_sample_range
                )

            # 5. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            variance = self._get_variance(timestep, prev_timestep)
            std_dev_t = eta * variance ** (0.5)

            if use_clipped_model_output:
                # the pred_epsilon is always re-derived from the clipped x_0 in Glide
                pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

            # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

            ## MPGD for editing
            def gradients(z, vae_model, feats2joints, control, mask_control, length):
                with torch.enable_grad():
                    z.requires_grad_(True)
                    x_ = vae_model.forward_decoder(z).contiguous()
                    
                    joint_pos = feats2joints(x_[:,:length,:-1])


                    loss_m = torch.norm((joint_pos - control) * mask_control, dim=-1, p=2)
                    loss = loss_m.sum()
                    grad = torch.autograd.grad([loss], [z])[0]

                    z.detach()
                return grad, loss
            

            
            if control is not None:
                # process for control
                control = control.clone().detach()

                mask_control = control.view(control.shape[0], control.shape[1], control.shape[2], 3) != 0
                
                control = control.view(control.shape[0], control.shape[1], 22, 3) * mask_control

                # joint id
                joint_ids = []
                for m in mask_control:
                    joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
                    joint_ids.append(joint_id)
            
            def calc_grad_scale(timestep, scale_weight, max_step):
                
                scale = - timestep*(timestep-max_step)/max_step/max_step*4*scale_weight #quadratic curve
                #scale = scale_weight # fixed step size case
                
                return scale

            grad, loss = gradients(pred_original_sample, vae_model, feats2joints, control, mask_control, length)
            rho = calc_grad_scale(timestep, edit_scale, max_step)
            pred_original_sample = pred_original_sample - rho * grad.detach()

            # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            if eta > 0:
                if variance_noise is not None and generator is not None:
                    raise ValueError(
                        "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                        " `variance_noise` stays `None`."
                    )

                if variance_noise is None:
                    variance_noise = randn_tensor(
                        model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                    )
                variance = std_dev_t * variance_noise

                prev_sample = prev_sample + variance

            # Time-traveling
            model_output = alpha_s ** (0.5) * prev_sample + (1-alpha_s) ** (0.5) * torch.randn_like(prev_sample)

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return prev_sample




# from https://github.com/Mael-zys/T2M-GPT/blob/main/models/resnet.py
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)
    

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
            
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        elif activation == "leakyrelu":
            self.activation1 = nn.LeakyReLU(0.01)
            self.activation2 = nn.LeakyReLU(0.01)
            
        

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)     


    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
            
        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x
    

class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)