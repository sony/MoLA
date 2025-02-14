import os 
import torch
import numpy as np
import json
import clip

import options.option_diffusion as option_diffusion
import models.vaesan as vaesan
import utils.utils_model as utils_model
import utils.eval_mola as eval_mola
from dataset import dataset_TM_eval
import models.t2m_latent_diffusion as latent_diffusion
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_diffusion.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
#clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='/apdcephfs_cq2/share_1290939/maelyszhang/.cache/clip')  # Must set jit=False for training
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False) 
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vaesan.HumanVAESAN(args,
                       args.latent_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = latent_diffusion.Text2Motion_LatentDiffusion(args,
        io_channels=args.latent_dim, 
        patch_size=1,
        embed_dim=args.clip_dim, #768
        cond_token_dim=0,
        project_cond_tokens=False, # True?
        global_cond_dim=512,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=args.num_depth,
        num_heads=args.num_head
        )


print ('loading checkpoint from {}'.format(args.resume_vae))
ckpt = torch.load(args.resume_vae, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_dit is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_dit))
    ckpt = torch.load(args.resume_dit, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()


fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
traj_50 = []
loc_50 = []
avg_err = []
traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
repeat_time = 20
        
for i in range(repeat_time):
    if args.edit_mode is None:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, logger = eval_mola.evaluation_stage2_test(args.out_dir, val_loader, net, trans_encoder, logger, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False, savegif=False, save=False, savenpy=(i==0))
    else:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, control_err, logger = eval_mola.evaluation_stage2_editing_test(args.out_dir, val_loader, net, trans_encoder, logger, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_multi=0, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False, savegif=False, save=False, savenpy=(i==0), edit_mode=args.edit_mode, edit_scale=args.edit_scale)
        traj_50.append(control_err[1])
        loc_50.append(control_err[3])
        avg_err.append(control_err[4])
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    multi.append(best_multi)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)
print('multi: ', sum(multi)/repeat_time)


fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
multi = np.array(multi)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)

if args.edit_mode is not None:
    msg_final_edit = f"traj_fail_50cm. {np.mean(traj_50):.3f}, conf. {np.std(traj_50)*1.96/np.sqrt(repeat_time):.3f}, kps_fail_50cm. {np.mean(loc_50):.3f}, conf. {np.std(loc_50)*1.96/np.sqrt(repeat_time):.3f}, kps_mean_err(m). {np.mean(avg_err):.3f}, conf. {np.std(avg_err)*1.96/np.sqrt(repeat_time):.3f}"
    logger.info(msg_final_edit)