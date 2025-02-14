import os
import json

import torch
import torch.optim as optim

import models.vaesan as vaesan
import utils.losses as losses 
import options.option_vaesan as option_vaesan
import utils.utils_model as utils_model
from dataset import dataset_VAESAN, dataset_TM_flag_eval
import utils.eval_mola as eval_mola
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer

import numpy as np

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_vaesan.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
train_loader = dataset_VAESAN.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

train_loader_iter = dataset_VAESAN.cycle(train_loader)

val_loader = dataset_TM_flag_eval.DATALoader(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t)

##### ---- Network ---- #####
net = vaesan.HumanVAESAN(args, ## use args to define different parameters in different quantizers
                       args.latent_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vae_act,
                       args.vae_norm)



if args.resume_vae : 
    logger.info('loading checkpoint from {}'.format(args.resume_vae))
    ckpt = torch.load(args.resume_vae, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
g_optimizer = optim.AdamW(net.vae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
g_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

d_optimizer = optim.AdamW(net.discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

Loss = losses.ReConsLossMask(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
avg_recons, avg_kl = 0., 0.
avg_gen, avg_disc = 0., 0.

for nb_iter in range(1, args.warm_up_iter):
    
    g_optimizer, current_lr = update_lr_warm_up(g_optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion = next(train_loader_iter)
    if np.isnan(gt_motion).any():
        continue
    gt_motion = gt_motion.cuda().float() # (bs, 64, dim)
    if args.decoder_output == 'except_rot':
        nb_joints = 21 if args.dataname == 'kit' else 22
        gt_motion = torch.cat((gt_motion[:, :, :(4+(nb_joints-1)*3)], gt_motion[:, :, (4+(nb_joints-1)*9):]), dim=2)

    pred_motion, loss_kl = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    
    #===============================
    # Generator training part
    #===============================

    vae_loss = loss_motion + args.kl_weight * loss_kl + args.loss_vel * loss_vel
    g_loss = args.gan_weight*net.adversarial(pred_motion, target='gen', flg_train=True)
    g_loss += vae_loss
    
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    #===============================
    # Discriminator training part
    #===============================

    d_loss = args.gan_weight*net.adversarial(gt_motion, target='real', flg_train=True)
    d_loss += args.gan_weight*net.adversarial(pred_motion.detach(), target='fake', flg_train=True)

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()


    avg_recons += loss_motion.item()
    avg_kl += loss_kl.item()

    avg_gen += g_loss.item()
    avg_disc += d_loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_kl /= args.print_iter
        
        avg_gen /= args.print_iter
        avg_disc /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t kl. {avg_kl:.5f} \t Recons.  {avg_recons:.5f} \t Gen_loss.  {avg_gen:.5f} \t Disc_loss.  {avg_disc:.5f}" )
        
        avg_recons, avg_kl = 0., 0.

##### ---- Training ---- #####
avg_recons, avg_kl = 0., 0.
best_mpjpe, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, logger = eval_mola.evaluation_vae(args.out_dir, val_loader, net, logger, 0, best_mpjpe=10000, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)

for nb_iter in range(1, args.total_iter + 1):
    
    gt_motion = next(train_loader_iter)
    if np.isnan(gt_motion).any():
        continue
    gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
    if args.decoder_output == 'except_rot':
        nb_joints = 21 if args.dataname == 'kit' else 22
        gt_motion = torch.cat((gt_motion[:, :, :(4+(nb_joints-1)*3)], gt_motion[:, :, (4+(nb_joints-1)*9):]), dim=2)
    
    pred_motion, loss_kl = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)

    #===============================
    # Generator training part
    #===============================
    
    vae_loss = loss_motion + args.kl_weight * loss_kl + args.loss_vel * loss_vel
    g_loss = args.gan_weight*net.adversarial(pred_motion, target='gen', flg_train=True)
    g_loss += vae_loss
    
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    g_scheduler.step()

    #===============================
    # Discriminator training part
    #===============================
    
    d_loss = args.gan_weight*net.adversarial(gt_motion, target='real', flg_train=True)
    d_loss += args.gan_weight*net.adversarial(pred_motion.detach(), target='fake', flg_train=True)

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    d_scheduler.step()

    avg_recons += loss_motion.item()
    avg_kl += loss_kl.item()

    avg_gen += g_loss.item()
    avg_disc += d_loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_kl /= args.print_iter

        avg_gen /= args.print_iter
        avg_disc /= args.print_iter
        
        
        logger.info(f"Train. Iter {nb_iter} : \t kl. {avg_kl:.5f} \t Recons.  {avg_recons:.5f} \t Gen_loss.  {avg_gen:.5f} \t Disc_loss.  {avg_disc:.5f}")
        
        avg_recons, avg_kl = 0., 0.,

    if nb_iter % args.eval_iter==0 :
        best_mpjpe, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, logger = eval_mola.evaluation_vae(args.out_dir, val_loader, net, logger, nb_iter, best_mpjpe, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        