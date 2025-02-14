import os
import json

import torch
import numpy as np
import models.vaesan as vaesan
import options.option_vaesan as option_vaesan
import utils.utils_model as utils_model
from dataset import dataset_TM_flag_eval
import utils.eval_mola as eval_mola
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
##### ---- Exp dirs ---- #####
args = option_vaesan.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
args.nb_joints = 21 if args.dataname == 'kit' else 22

val_loader = dataset_TM_flag_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2**args.down_t)

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

mpjpe = []
fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
repeat_time = 20
for i in range(repeat_time):
    best_mpjpe, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, logger = eval_mola.evaluation_vae(args.out_dir, val_loader, net, logger, 0, best_mpjpe=10000, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=False, save=False, savenpy=(i==0))
    mpjpe.append(best_mpjpe)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
print('final result:')
print('mpjpe: ', sum(mpjpe)/repeat_time)
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

mpjpe = np.array(mpjpe)
fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"MPJPE. {np.mean(mpjpe):.3f}, FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)