import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


#Based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
def _step_with_mpgd(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    max_step: int,
    repeat: int,
    scale_weight: float,
    sample: torch.FloatTensor,
    vae_model,
    feats2joints,
    control, 
    lengths,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
    return_dict: bool = True,
):

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
        def gradients(z, vae_model, feats2joints, control, mask_control, joint_ids=None):
            with torch.enable_grad():
                z.requires_grad_(True)

                x_ = vae_model.decode(z, lengths).contiguous()
                joint_pos = feats2joints(x_)

                loss = torch.norm((joint_pos - control) * mask_control, dim=-1, p=2)
                grad = torch.autograd.grad([loss.sum()], [z])[0]
                
                z.detach()
            return loss, grad
        
        def calc_grad_scale(timestep, scale_weight):
            
            scale = - timestep*(timestep-max_step)/max_step/max_step*4*scale_weight #quadratic curve
            #scale = 0.04 # fixed step size
            
            return scale
        
        # process for control
        control = control.clone().detach()

        mask_control = control.view(control.shape[0], control.shape[1], control.shape[2], 3) != 0
        
        control = control.view(control.shape[0], control.shape[1], 22, 3) * mask_control

        # joint id
        joint_ids = []
        for m in mask_control:
            joint_id = torch.nonzero(m.sum(0).squeeze(-1) != 0).squeeze(1)
            joint_ids.append(joint_id)
        
        #pred_original_sample = pred_original_sample.detach().requires_grad_(True)
        
        _, grad = gradients(pred_original_sample, vae_model, feats2joints, control, mask_control, joint_ids)
        rho = calc_grad_scale(timestep, scale_weight) / (grad*grad).mean().sqrt().item()
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
        model_output = alpha_s ** (0.5) * prev_sample + (1 - alpha_s) ** (0.5) * torch.randn_like(prev_sample)

    if not return_dict:
        return (prev_sample,)

    return prev_sample



def _normalize(tensor, dim):
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min(1e-12)
    return tensor / denom


# Based on https://github.com/sony/san/blob/main/stylesan-xl/pg_modules/san_modules.py
class SANConv1d(nn.Conv1d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None
                 ):
        super(SANConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding=padding)
        scale = self.weight.norm(p=2.0, dim=[1, 2], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(in_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, flg_train=False):
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1)
        if flg_train:
            our_fun = F.conv1d(input, normalized_weight.detach(), None, self.stride,
                               self.padding, self.dilation, self.groups)
            our_dir = F.conv1d(input.detach(), normalized_weight, None, self.stride,
                               self.padding, self.dilation, self.groups)
            out = [our_fun * scale, our_dir * scale.detach()]
        else:
            out = F.conv1d(input, normalized_weight, None, self.stride,
                           self.padding, self.dilation, self.groups)
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2])