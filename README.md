# MoLA: Motion Generation and Editing with Latent Diffusion Enhanced by Adversarial Training

This repository is the official implementation of "MoLA: Motion Generation and Editing with Latent Diffusion Enhanced by Adversarial Training"

> **Abstract:** In text-to-motion generation, controllability as well as generation quality and speed has become increasingly critical. The controllability challenges include generating a motion of a length that matches the given textual description and editing the generated motions according to control signals, such as the start-end positions and the pelvis trajectory. In this paper, we propose MoLA, which provides fast, high-quality, variable-length motion generation and can also deal with multiple editing tasks in a single framework. Our approach revisits the motion representation used as inputs and outputs in the model, incorporating an activation variable to enable variable-length motion generation. Additionally, we integrate a variational autoencoder and a latent diffusion model, further enhanced through adversarial training, to achieve high-quality and fast generation. Moreover, we apply a training-free guided generation framework to achieve various editing tasks with motion control inputs. We quantitatively show the effectiveness of adversarial learning in text-to-motion generation, and demonstrate the applicability of our editing framework to multiple editing tasks in the motion domain.

- Paper: [arXiv](https://arxiv.org/abs/2406.01867)
- Demo: [Project page](https://kengouchida-sony.github.io/MoLA-demo)
- Pretrained model: [Hugging Face](https://huggingface.co/Sony/MoLA)


## Update
- [2025/2/14] Update architecture and training code (Release MoLA-v2)
- [2024/6/12] First release

## Setup

#### Environment
Install the packages in `requirements.txt`.
```shell
$ pip install -r requirements.txt
```


#### Dependenicies
Run the script to download the dependency materials:

```shell
$ bash dataset/prepare/download_glove.sh
$ bash dataset/prepare/download_extractor.sh
```


#### Dataset
Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup. Copy the result dataset to our repository:
```shell
$ cp -r ../HumanML3D/HumanML3D ./datasets/humanml3d
```

## Training

#### Traning Stage 1 (VAE+SAN):

Run the following script:
```shell
$ python train_vaesan.py --batch-size 128 --latent-dim 16 --out-dir output --encoder-input root_pos_rot --vae-act leakyrelu --kl-weight 1e-4 --gan-weight 1e-3 --exp-name mola_stage1
```

#### Training Stage 2(Conditional motion latent diffusion):

Run the following script:
```shell
$ python train_diffusion.py --batch-size 64 --out-dir output --resume-vae output/mola_stage1/net_best_fid.pth --latent-dim 16 --clip-dim 512 --encoder-input root_pos_rot --cfg-guidance-scale 11 --exp-name mola_stage2
```

## Evaluation

#### Stage 1 (Reconstruction performance):
To evaluate reconstruction performance of stage1 model, run the following command:
```shell
$ python eval_stage1_vaesan.py --batch-size 32 --out-dir output --vae-act leakyrelu --encoder-input root_pos_rot --resume-vae output/mola_stage1/net_best_fid.pth --exp-name test_mola_stage1
```

#### Stage 2 (Text-conditional generation performance):
To evaluate motion generation performance of stage2 model, run the following command:
```shell
$ python eval_stage2_diffusion.py --batch-size 32 --latent-dim 16 --resume-vae output/mola_stage1/net_best_fid.pth --resume-dit output/mola_stage2/net_best_fid.pth --vae-name VAESAN --out-dir output --encoder-input root_pos_rot --vae-act leakyrelu --inference-timestep 50 --cfg-guidance-scale 11 --exp-name test_mola_stage2
```

#### Motion editing:
To evaluate motion editing (on path-following task) performance, run the following command:
```shell
$ python eval_stage2_diffusion.py --batch-size 32 --latent-dim 16 --resume-vae output/mola_stage1/net_best_fid.pth --resume-dit output/mola_stage2/net_best_fid.pth --vae-name VAESAN --out-dir output --encoder-input root_pos_rot --vae-act leakyrelu --inference-timestep 50 --cfg-guidance-scale 11 --exp-name test_mola_stage2 --edit-mode path
```


## Visualizing generated samples

We support text prompt (for text-to-motion) and npy file(for control signal on motion editing) as input.
The generated/edited motions are npy files.

### Text-to-Motion
Our model generates motion based on a specified text prompt (`--prompt`) using pre-trained parameters (stage1: `--resume-vae`, stage2: `--resume-dit`).
```shell
$ python visualize_test.py --resume-vae pretrained/stage1.pth --resume-dit pretrained/stage2.pth--cfg-guidance-scale 11 --prompt "A person walks to with their hands up."
```


### Motion editing
Our model achieves multiple editing tasks, including path-following (`--edit-mode=path`), in-betweening (`--edit-mode=inbetweening`), and upper-body editing (`--edit-mode=upper_edit`), within the same framework.
```shell
$ python visualize_test.py --resume-vae pretrained/stage1.pth --resume-dit pretrained/stage2.pth --cfg-guidance-scale 11 --prompt "A person walks to with their hands up." --edit-mode inbetweening --edit-scale 0.1
```



## Citation
```bibtex
@article{uchida2024mola,
        title={MoLA: Motion Generation and Editing with Latent Diffusion Enhanced by Adversarial Training},
        author={Uchida, Kengo and Shibuya, Takashi and Takida, Yuhta and Murata, Naoki and Tanke, Julian and Takahashi, Shusuke and Mitsufuji, Yuki},
        journal={arXiv preprint arXiv:2406.01867},
        year={2024}
      }
```

## Reference
Part of the code is borrowed from the following repositories. 
We would like to thank the authors of these repos for their excellent work: 
[HumanML3D](https://github.com/EricGuo5513/HumanML3D),
[MPGD](https://github.com/KellyYutongHe/mpgd_pytorch/),
[SAN](https://github.com/sony/san),
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT),
[MLD](https://github.com/ChenFengYe/motion-latent-diffusion),
[Stable-Audio](https://github.com/Stability-AI/stable-audio-tools).
