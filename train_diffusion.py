import os 
import torch
from torch.nn import functional as F
import numpy as np

from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip

import options.option_diffusion as option_diffusion
import models.vaesan as vaesan
import utils.utils_model as utils_model
import utils.eval_mola as eval_mola
from dataset import dataset_DiT_train
from dataset import dataset_TM_eval
from dataset import dataset_latent_token
import models.t2m_latent_diffusion as latent_diffusion
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

from utils.lr import CosineAnnealingLR, InverseLR


##### ---- Exp dirs ---- #####
args = option_diffusion.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vae_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vae_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vae_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader_token = dataset_latent_token.DATALoader(args.dataname, 1, unit_length=2**args.down_t)

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

stage1_model = vaesan.HumanVAESAN(args,
                       args.latent_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


stage2_model = latent_diffusion.Text2Motion_LatentDiffusion(args,
        io_channels=args.latent_dim, 
        patch_size=1,
        embed_dim=args.clip_dim,
        cond_token_dim=0,
        project_cond_tokens=False,
        global_cond_dim=512,
        project_global_cond=False,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=args.num_depth,
        num_heads=args.num_head
        )

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

print(f"Stage1 trainable parameters: {count_parameters(stage1_model)}")
print(f"Stage2 trainable parameters: {count_parameters(stage2_model)}")
print ('loading checkpoint from {}'.format(args.resume_vae))
ckpt = torch.load(args.resume_vae, map_location='cpu')
stage1_model.load_state_dict(ckpt['net'], strict=True)
stage1_model.eval()
stage1_model.cuda()

if args.resume_dit is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_dit))
    ckpt = torch.load(args.resume_dit, map_location='cpu')
    stage2_model.load_state_dict(ckpt['stage2_model'], strict=True)
stage2_model.train()
stage2_model.cuda()

##### ---- Optimizer & Scheduler ---- #####
from torch.optim import AdamW
optimizer = AdamW(lr=args.lr, params=stage2_model.parameters(), betas=(0.9, 0.99))
scheduler = CosineAnnealingLR(optimizer, max_epochs=args.total_iter, warmup_epochs=args.warm_up_epochs, warmup_start_lr=1.0e-8, eta_min=1.0e-6)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()
loss_dif = torch.nn.MSELoss(reduction='mean')

nb_iter, avg_loss_diffusion, avg_acc = 0, 0., 0.
right_num = 0
#nb_sample_train = 0

##### ---- get code ---- #####
for batch in train_loader_token:
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    z = stage1_model.encode(pose)
    z = z.cpu().detach().numpy()
    np.save(pjoin(args.vae_dir, name[0] +'.npy'), z) 


train_loader = dataset_DiT_train.DATALoader(args.dataname, args.batch_size, args.latent_dim, args.vae_name, unit_length=2**args.down_t)
train_loader_iter = dataset_DiT_train.cycle(train_loader)


        
##### ---- Training ---- #####
best_fid=1000
best_iter=0
best_div=100
best_top1=0
best_top2=0
best_top3=0
best_matching=100
while nb_iter <= args.total_iter:
    
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    
    m_tokens = m_tokens.cuda().float()
    bs = m_tokens.shape[0]
    target = m_tokens[..., :m_tokens_len.max().item()]
    target = target.cuda()
    
    
    text = clip.tokenize(clip_text, truncate=True).cuda()
    
    feat_clip_text = clip_model.encode_text(text).float()

    noise, noise_pred = stage2_model(target, feat_clip_text, m_tokens_len)

    loss_diffusion = loss_dif(noise, noise_pred)

    ## global loss
    optimizer.zero_grad()
    loss_diffusion.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_diffusion = avg_loss_diffusion + loss_diffusion.item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_diffusion = avg_loss_diffusion / args.print_iter
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_diffusion:.5f}"
        logger.info(msg)
        avg_loss_diffusion = 0.
        right_num = 0

    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, logger = eval_mola.evaluation_stage2(args.out_dir, val_loader, stage1_model, stage2_model, logger, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper)

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            