# This code is based on https://github.com/ChenFengYe/motion-latent-diffusion under the MIT license.

from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from mola.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mola.models.operator import PositionalEncoding
from mola.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from mola.models.operator.position_encoding import build_position_encoding
from mola.utils.temos_utils import lengths_to_mask

from .mld_vae import MldVae

class MoLAVAEGAN(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = ablation.MLP_DIST
        self.pe_type = ablation.PE_TYPE

        self.vae = MldVae(ablation, nfeats, latent_dim, ff_size, num_layers, num_heads, dropout, arch, normalize_before, activation, position_embedding, **kwargs)

        #add for GAN
        self.discriminator = GANDiscriminator(nfeats, output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=1, activation='leakyrelu', norm=None)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.vae.encode(features, lengths)
        feats_rst = self.vae.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.vae.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.vae.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        if self.vae.pe_type == "actor":
            xseq = self.vae.query_pos_encoder(xseq)
            dist = self.vae.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.vae.pe_type == "mld":
            xseq = self.vae.query_pos_encoder(xseq)
            dist = self.vae.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]


        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.vae.mlp_dist:
            tokens_dist = self.vae.dist_layer(dist)
            mu = tokens_dist[:, :, :self.vae.latent_dim]
            logvar = tokens_dist[:, :, self.vae.latent_dim:]
        else:
            mu = dist[0:self.vae.latent_size, ...]
            logvar = dist[self.vae.latent_size:, ...]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.vae.latent_dim, device=z.device)


        if self.vae.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.vae.latent_size),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.vae.pe_type == "actor":
                xseq = self.vae.query_pos_decoder(xseq)
                output = self.vae.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.vae.pe_type == "mld":
                xseq = self.vae.query_pos_decoder(xseq)
                output = self.vae.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]

        elif self.vae.arch == "encoder_decoder":
            if self.vae.pe_type == "actor":
                queries = self.vae.query_pos_decoder(queries)
                output = self.vae.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask).squeeze(0)
            elif self.vae.pe_type == "mld":
                queries = self.vae.query_pos_decoder(queries)
                output = self.vae.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                ).squeeze(0)


        output = self.vae.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats

    def adversarial(
            self,
            features: Tensor,
            target: str,
            lengths: Optional[List[int]] = None,
    ):
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        #x = self.skel_embedding(x)

        adv_loss = self.discriminator(x, target)
        return adv_loss

class GANDiscriminator(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='leakyrelu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.LeakyReLU(0.01))
        
        for i in range(down_t):
            input_dim = width
            #if i == 0:
                #stride_t, width = 1, 1024
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D_disc(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x
    
    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x
    
    def feature(self, x):
        x_in = self.preprocess(x)
        
        return self.model(x_in)

    def forward(self, x, target):
        x_in = self.preprocess(x)
        logits = self.model(x_in)

     
        if target == "gen":
            adversarial_loss = -torch.mean(logits)   
        elif target == 'real':
            minval = torch.min(logits - 1, torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2]).to(logits.device))
            adversarial_loss = -torch.mean(minval)
        else:
            minval = torch.min(-logits - 1, torch.zeros(logits.shape[0], logits.shape[1], logits.shape[2]).to(logits.device))
            adversarial_loss = -torch.mean(minval)

        return adversarial_loss

class Resnet1D_disc(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()
        
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate, activation=activation, norm=norm) for depth in range(n_depth)]
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