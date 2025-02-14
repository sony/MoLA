import torch
import torch.nn as nn
import torch.nn.functional as F

def _normalize(tensor, dim):
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min(1e-12)
    return tensor / denom


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

