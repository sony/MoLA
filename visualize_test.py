
import os
import time

from pathlib import Path

import numpy as np
import torch

from mola.config import parse_args
from mola.data.get_data import get_datasets
from mola.models.get_model import get_model
from mola.utils.logger import create_logger


def main():
    """
    tasks:
         1. standard text-to-motion generation
         2. motion editing with guided generation
    """
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")


    from mola.utils.demo_utils import load_example_input
    text, length = load_example_input(cfg.DEMO.EXAMPLE)


    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create MoLA model
    total_time = time.time()
    model = get_model(cfg, dataset)


    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]


    model.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    # sample
    with torch.no_grad():

        # task: input or Example
        if text:
            # prepare batch data
            batch = {"length": length, "text": text}
            
            for rep in range(cfg.DEMO.REPLICATION):
                # motion editing
                if cfg.DEMO.EDITING:
                    #prep. for control signal
                    control_joints= torch.Tensor(np.load(cfg.DEMO.CONTROL)).unsqueeze(0)
                    control = control_joints.cuda()
                    joints = model.edit_with_mpgd(batch, control)
                    
                # text-to-motion generation
                else:
                    joints = model(batch)

                nsample = len(joints)
                id = 0
                for i in range(nsample):
                    npypath = str(output_dir /
                                f"len{length[i]}_batch{id}_{i}.npy")
                    with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                        text_file.write(batch["text"][i])
                    
                    if cfg.DEMO.EDITING:
                        control_path = os.path.splitext(npypath)[0] + '_control' + os.path.splitext(npypath)[1]
                        control_gen_path = os.path.splitext(npypath)[0] + '_' + cfg.DEMO.EDIT_TYPE + os.path.splitext(npypath)[1]
                        np.save(control_path, control[i].detach().cpu().numpy())
                        np.save(control_gen_path, joints[i].detach().cpu().numpy())
                    
                    np.save(npypath, joints[i].detach().cpu().numpy())
                    logger.info(f"Motions are generated here:\n{npypath}")

        total_time = time.time() - total_time
        print(
            f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
        )
        
    if cfg.DEMO.VISUALIZE:
        # plot bone with lines
        from mola.data.humanml.utils.plot_script import plot_3d_motion, plot_3d_condition
        for i in range(len(text)):
            
            j_data = joints[i].to('cpu').detach().numpy().copy()

            if cfg.DEMO.EDITING:

                control_data = control[i].to('cpu').detach().numpy().copy()
                fig_control_path = Path(str(control_path).replace(".npy",".gif"))
                if cfg.DEMO.EDIT_TYPE == 'inbetweening':
                    control_txt = 'start-end control'
                elif cfg.DEMO.EDIT_TYPE == 'upper':
                    control_txt = 'lower-body control'
                elif cfg.DEMO.EDIT_TYPE == 'path':
                    control_txt = 'path control'
                plot_3d_condition(fig_control_path, control_data, j_data, title=control_txt, fps=cfg.DEMO.FRAME_RATE, edit_type=cfg.DEMO.EDIT_TYPE, plot_type='control_only')
                fig_control_gen_path = Path(str(control_gen_path).replace(".npy",".gif"))
                plot_3d_condition(fig_control_gen_path, control_data, j_data, title=text[i], fps=cfg.DEMO.FRAME_RATE, edit_type=cfg.DEMO.EDIT_TYPE, plot_type='control_gen')
            
            else:
                npypath = str(output_dir /
                        f"len{length[i]}_batch{id}_{i}_{rep}.npy")
                fig_path = Path(str(npypath).replace(".npy",".gif"))
                plot_3d_motion(fig_path, j_data, title=text[i], fps=cfg.DEMO.FRAME_RATE)
            

if __name__ == "__main__":
    main()
