import torch.nn as nn

import torch
from models.resnet import ResConv1DBlock
from models.san_modules import SANConv1d

from models.resnet import Resnet1D

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 latent_width = 32,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        if activation == 'leakyrelu':
            blocks.append(nn.LeakyReLU(0.01))
        else:
            blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

        self.final_mu = nn.Conv1d(width, latent_width, 3, 1, 1)
        self.final_logvar = nn.Conv1d(width, latent_width, 3, 1, 1)

    def forward(self, x):
        x = self.model(x)
        mu = self.final_mu(x)
        logvar = self.final_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 latent_width = 32,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(latent_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class VAE(nn.Module):
    def __init__(self,
                 args,
                 latent_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 encoder_input='all',
                 decoder_output='all'):

        super().__init__()

        if encoder_input == "all":
            self.encoder_motion_dim = 252 if args.dataname == 'kit' else 264
        elif encoder_input == "root_pos_rot":
            self.encoder_motion_dim = 185 if args.dataname == 'kit' else 194
        elif encoder_input == "root_pos":
            self.encoder_motion_dim = 65 if args.dataname == 'kit' else 68
        if decoder_output == "all":
            self.decoder_motion_dim = 252 if args.dataname == 'kit' else 264
        elif decoder_output == "except_rot":
            self.decoder_motion_dim = 132 if args.dataname == 'kit' else 138

        self.encoder = Encoder(self.encoder_motion_dim, output_emb_width, latent_dim,  down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(self.decoder_motion_dim, output_emb_width, latent_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

        self.klloss = KLLoss()


    def preprocess(self, x_in):
        # Clipping only the information necessary
        x = torch.cat((x_in[:, :, :self.encoder_motion_dim-1], x_in[:, :, -1:]), dim=2) 
        
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        x_in = self.preprocess(x)
        mu, logvar = self.encoder(x_in)
        std = logvar.exp().pow(0.5)
        z =  mu + torch.randn(mu.shape, device=mu.device) * std
        #dist_m = torch.distributions.Normal(mu, std)
        #z = dist_m.rsample()

        return z
    

    def forward(self, x):

        x_in = self.preprocess(x)
        
        # Encode
        mu, logvar = self.encoder(x_in)


        std = logvar.exp().pow(0.5)
        dist_m = torch.distributions.Normal(mu, std)

        mu_ref = torch.zeros_like(dist_m.loc)
        scale_ref = torch.ones_like(dist_m.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        

        loss = self.klloss(dist_m, dist_ref)

        # decoder
        z =  mu + torch.randn(mu.shape, device=mu.device) * std
        #z = dist_m.rsample()
        x_decoder = self.decoder(z)

        x_out = self.postprocess(x_decoder)


        # activation in [0, 1]
        sigmoid = nn.Sigmoid()
        x_out[...,  -1] = sigmoid(x_out[..., -1])

        return x_out, loss


    def forward_decoder(self, z):

        # decoder
        x_decoder = self.decoder(z)
        x_out = self.postprocess(x_decoder)

        return x_out
    
    def forward_decoder_clip(self, z, length=None):
        assert z.shape[0] == 1

        # decoder
        x_decoder = self.decoder(z)
        x_pose = self.postprocess(x_decoder)

        # activation in [0, 1]
        sigmoid = nn.Sigmoid()
        x_pose[...,  -1] = sigmoid(x_pose[..., -1])

        flag = torch.abs(x_pose[0,:,-1]) > 0.999
        false_indices = torch.nonzero(flag == False)
        flag_idx = false_indices[0] if len(false_indices) > 0 else len(flag) - 1
      
        flag_idx = max(flag_idx, 10)

        if length is None:
            x_out = x_pose[:,:flag_idx,:-1]
        else:
            # TODO
            x_out = x_pose[:,:length,:-1]

        return x_out


class HumanVAESAN(nn.Module):
    def __init__(self,
                 args,
                 latent_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()

        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vae = VAE(args, latent_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, encoder_input=args.encoder_input, decoder_output=args.decoder_output)

        if args.decoder_output == "all":
            disc_input_dim = 252 if args.dataname == 'kit' else 264
        elif args.decoder_output == "except_rot":
            disc_input_dim = 132 if args.dataname == 'kit' else 138
        self.discriminator = SANDiscriminator(disc_input_dim, output_emb_width, down_t, stride_t, width, depth=3, dilation_growth_rate=1, activation="leakyrelu", norm=norm)
        
    def encode(self, x):
        b, t, c = x.size()
        z = self.vae.encode(x) # (N, T)
        return z

    def forward(self, x):

        x_out, loss = self.vae(x)

        return x_out, loss

    def forward_decoder(self, x):
        x_out = self.vae.forward_decoder(x)
        return x_out
    
    def forward_decoder_clip(self, x, length=None):
        x_out = self.vae.forward_decoder_clip(x, length=length)
        return x_out

    def adversarial(self, x, target, flg_train=False):
        loss = self.discriminator(x, target, flg_train=flg_train)
        return loss

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"

    

class SANDiscriminator(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D_disc(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        #blocks.append(SANConv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        #self.fc = nn.Linear(output_emb_width*16, 1, True)
        self.last_layer = SANConv1d(width, output_emb_width, 3, 1, 1)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x

    def forward(self, x, target, flg_train=False):
        x_in = self.preprocess(x)
        prediction = self.model(x_in)
        logits_fun, logits_dir = self.last_layer(prediction, flg_train=flg_train)
        #prediction = prediction.view(prediction.size(0), -1)
        #logits = self.fc(prediction)

        # for h
        if target == "gen":
            adversarial_fun_loss = -torch.mean(logits_fun)
        elif target == 'real':
            minval = torch.min(logits_fun - 1, torch.zeros(logits_fun.shape[0], 512, 16).to(logits_fun.device))
            adversarial_fun_loss = -torch.mean(minval)
        else:
            minval = torch.min(-logits_fun - 1, torch.zeros(logits_fun.shape[0], 512, 16).to(logits_fun.device))
            adversarial_fun_loss = -torch.mean(minval)

        # for w
        if target == "gen":
            adversarial_dir_loss = -torch.mean(logits_dir)
        elif target == 'real':
            adversarial_dir_loss = -torch.mean(logits_dir)
        else:
            adversarial_dir_loss =  torch.mean(logits_dir)


        adversarial_loss = adversarial_fun_loss + adversarial_dir_loss
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