import os
import clip
import torch
import numpy as np
import models.vaesan as vaesan
import models.t2m_latent_diffusion as latent_diffusion

from utils.motion_process import recover_from_ric
import visualization.plot_3d_global as plot_3d
import options.option_diffusion as option_diffusion

args = option_diffusion.get_args_parser()
device = torch.device('cuda')



## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device(device), jit=False, download_root='./')  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vaesan.HumanVAESAN(args, ## use args to define different parameters in different quantizers
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
        depth=9,
        num_heads=4
        )


print ('loading checkpoint from {}'.format(args.resume_vae))
ckpt = torch.load(args.resume_vae, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda(device)

print ('loading transformer checkpoint from {}'.format(args.resume_dit))
ckpt = torch.load(args.resume_dit, map_location='cpu')
stage2_model.load_state_dict(ckpt['trans'], strict=True)
stage2_model.eval()
stage2_model.cuda(device)


mean = torch.from_numpy(np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')).cuda()

#visualization
clip_text = [args.prompt]
#clip_text = ["A person walks to with their hands up."]

text = clip.tokenize(clip_text, truncate=True).cuda(device)
feat_clip_text = clip_model.encode_text(text).float()

if args.edit_mode == 'path':
    print("EDIT:" + args.edit_mode)
    control_joints= torch.Tensor(np.load("./demo/control_example_path_zigzag.npy")).unsqueeze(0)
    control = control_joints.cuda()
elif args.edit_mode == 'inbetweening':
    print("EDIT:" + args.edit_mode)
    control_joints= torch.Tensor(np.load("./demo/control_example_start_end_left.npy")).unsqueeze(0)
    control = control_joints.cuda()
elif args.edit_mode == 'upper_edit':
    print("EDIT:" + args.edit_mode)
    control_joints= torch.Tensor(np.load("./demo/control_example_lowerbody_circle.npy")).unsqueeze(0)
    control = control_joints.cuda()



if args.edit_mode is not None:
    edit_scale = args.edit_scale
    z = stage2_model._diffusion_reverse(feat_clip_text, lengths=control.shape[1], control=control.detach(), stage1_model=net, edit_scale=edit_scale)
else:
    z = stage2_model._diffusion_reverse(feat_clip_text, lengths=None) 


pred_pose = net.forward_decoder_clip(z)


pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22) #if kit:21, else humanml3d:22
xyz = pred_xyz.reshape(1, -1, 22, 3) #if kit:21, else humanml3d:22



if args.edit_mode is not None:
    np.save(os.path.join('./output', 'visualize_test_'+ args.edit_mode + '.npy'), xyz.detach().squeeze(0).cpu().numpy())
    pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), clip_text, ['output/visualize_test_'+ args.edit_mode + '.gif'], control.squeeze(0).cpu().numpy(), args.edit_mode, 'control_gen')
else:
    np.save(os.path.join('./output', 'visualize_test.npy'), xyz.detach().squeeze(0).cpu().numpy())
    pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), clip_text, ['output/visualize_test.gif'])