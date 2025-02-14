import os

import clip
import numpy as np
import torch
from scipy import linalg

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from utils.motion_process import feats2joints

import time


@torch.no_grad()        
def evaluation_vae(out_dir, val_loader, net, logger, nb_iter, best_mpjpe, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    mpjpe = 0
    num_poses  = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
        if np.isnan(motion).any():
            continue

        motion = motion.cuda()
        dim_motion = motion.shape[-1] - 1

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion[:,:,:dim_motion], m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] - 1 == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :dim_motion].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            pred_pose, loss_commit = net(motion[i:i+1, :m_length[i]])
            if pred_pose.shape[2] in [132, 138]:
                pred_j = feats2joints(pred_pose[..., :-1], num_joints, no_rot=True)
                rot_f = get_rot_feature(pred_j).cuda()
                pred_pose = torch.cat((pred_pose[:, :, :(4+(num_joints-1)*3)], rot_f, pred_pose[:, :, (4+(num_joints-1)*3):]), dim=2)
            pred_denorm = val_loader.dataset.inv_transform(pred_pose[:,:,:dim_motion].detach().cpu().numpy())
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            if savenpy:
                filename = os.path.join(out_dir, name[i]+'_gt.npy')
                os.makedirs(os.path.dirname(filename), exist_ok = True)
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])


        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval[:,:,:dim_motion], m_length)

        # to calculate mpjpe
        bgt = val_loader.dataset.inv_transform(motion[:,:,:dim_motion].detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval[:,:,:dim_motion].detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joints)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joints)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    mpjpe = mpjpe.cpu().numpy() / num_poses


    msg = f"--> \t Eva. Iter {nb_iter} :, MPJPE. {mpjpe:.4f}, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    

    if mpjpe < best_mpjpe :
        msg = f"--> --> \t MPJPE Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
        logger.info(msg)
        best_mpjpe = mpjpe

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_mpjpe, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, logger


@torch.no_grad()        
def evaluation_stage2(out_dir, val_loader, net, trans, logger, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    times = []

    for i in range(1):
        for batch in val_loader:
            times = []
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

            if np.isnan(pose).any():
                continue

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                starttime = time.time()
                z_motion = trans._diffusion_reverse(feat_clip_text[k:k+1], lengths=None)
                
                pred_pose = net.forward_decoder_clip(z_motion)
                cur_len = pred_pose.shape[1]
                
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])
                endtime = time.time()
                elapsed = endtime - starttime
                times.append(elapsed)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # additional evaluation
    #mpjpe = calculate_mpjpe(motion_annotation_np, motion_pred_np)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    
    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, logger


@torch.no_grad()        
def evaluation_stage2_test(out_dir, val_loader, net, trans, logger, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 
    print("calc_start")
    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    

    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        if np.isnan(pose).any():
            continue
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            starttime = time.time()
            times = []
            for k in range(bs):
                z_motion = trans._diffusion_reverse(feat_clip_text[k:k+1], lengths=None)
                
                pred_pose = net.forward_decoder_clip(z_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])
            endtime = time.time()
            elapsed = endtime - starttime
            times.append(elapsed)
            print(f"Average time per sample ({bs*len(times)}): ", np.mean(times)/bs)
            

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs
        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)


    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, logger



@torch.no_grad()        
def evaluation_stage2_editing_test(out_dir, val_loader, net, trans, logger, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, edit_mode='inbetweening', edit_scale=0.1) : 
    print("edit_mode: " + edit_mode)
    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        if np.isnan(pose).any():
            continue
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []

        from utils.motion_process import feats2joints, remove_padding
        hint_joints = feats2joints(pose.detach().float())
        hint_joints = remove_padding(hint_joints, m_length)

        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            starttime = time.time()
            times = []
            dist_sum = 0
            traj_err = []
            traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]

            for k in range(bs):
                hint = hint_joints[k].unsqueeze(0).cuda()
                if edit_mode == 'path':
                    for j_id in range(1,22):
                        hint[:, :, j_id, :] = 0
                else:
                    raise NotImplementedError()

                mask_hint = hint.view(hint.shape[0], hint.shape[1], num_joints, 3).sum(dim=-1, keepdim=True) != 0
                hint = hint.view(hint.shape[0], hint.shape[1], num_joints, 3) * mask_hint
                z_motion = trans._diffusion_reverse(feat_clip_text[k:k+1], lengths=hint.shape[1], control=hint, stage1_model=net, edit_scale=edit_scale)
                
                pred_pose = net.forward_decoder_clip(z_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                # for control
                pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
                for motion, h, mask in zip(pred_xyz , hint, mask_hint):
                    eval_len = min(motion.shape[0], h.shape[0])
                    control_error = control_l2(motion[:eval_len, :, :].unsqueeze(0).to('cpu').detach().numpy(), h[:eval_len, :, :].unsqueeze(0).to('cpu').detach().numpy(), mask[:eval_len,:,:].unsqueeze(0).to('cpu').detach().numpy())
                    mean_error = control_error.sum() / mask.sum()
                    dist_sum += mean_error
                    control_error = control_error.reshape(-1)
                    mask = mask.reshape(-1)
                    err_np = calculate_trajectory_error(control_error, mean_error, mask.cpu().detach().numpy())
                    traj_err.append(err_np)

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])
            endtime = time.time()
            elapsed = endtime - starttime
            times.append(elapsed)
            print(f"Average time per sample ({bs*len(times)}): ", np.mean(times)/bs)
            

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            #for trajectory error
            traj_err = np.stack(traj_err).mean(0)

            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs
        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)


    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)

    msg_control = f'---> \t Trajectory Error: '
    edit_err = []
    for (k, v) in zip(traj_err_key, traj_err):
        edit_err.append(np.mean(v))
        msg_control += '(%s): %.4f ' % (k, np.mean(v))
    logger.info(msg_control)
    

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, edit_err, logger



# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_trajectory_error(dist_error, mean_err_traj, mask, strict=True):
    ''' dist_error shape [5]: error for each kps in metre
      Two threshold: 20 cm and 50 cm.
    If mean error in sequence is more then the threshold, fails
    return: traj_fail(0.2), traj_fail(0.5), all_kps_fail(0.2), all_kps_fail(0.5), all_mean_err.
        Every metrics are already averaged.
    '''
    # mean_err_traj = dist_error.mean(1)
    if strict:
        # Traj fails if any of the key frame fails
        traj_fail_02 = 1.0 - (dist_error <= 0.2).all()
        traj_fail_05 = 1.0 - (dist_error <= 0.5).all()
    else:
        # Traj fails if the mean error of all keyframes more than the threshold
        traj_fail_02 = (mean_err_traj > 0.2)
        traj_fail_05 = (mean_err_traj > 0.5)
    all_fail_02 = (dist_error > 0.2).sum() / mask.sum()
    all_fail_05 = (dist_error > 0.5).sum() / mask.sum()

    return np.array([traj_fail_02, traj_fail_05, all_fail_02, all_fail_05, dist_error.sum() / mask.sum()])

def control_l2(motion, hint, hint_mask):
    # motion: b, seq, 22, 3
    # hint: b, seq, 22, 1
    loss = np.linalg.norm((motion - hint) * hint_mask, axis=-1)
    return loss

def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist