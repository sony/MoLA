# MoLA: Motion Generation and Editing with Latent Diffusion Enhanced by Adversarial Training

This repository is the official implementation of "MoLA: Motion Generation and Editing with Latent Diffusion Enhanced by Adversarial Training"


<p align="center">
  <a href='https://arxiv.org/pdf/2406.01867'>
  <img src='https://img.shields.io/badge/Paper-PDF-magenta?style=flat&logo=arXiv&logoColor=magenta'></a> 
  <a href='https://arxiv.org/abs/2406.01867'>
  <img src='https://img.shields.io/badge/Arxiv-2406.01867-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://k5uchida.github.io/MoLA-demo/'>
  <img src='https://img.shields.io/badge/Project-Page-yellow?style=flat&logo=Google%20chrome&logoColor=yellow'></a> 
</p>



## Setup

#### Environment
Install the packages in `requirements.txt`.
```shell
pip install -r requirements.txt
```

We test our code on Python 3.8.10 and PyTorch 2.1.0

#### Dependenicies
Run the script to download dependencies materials:

```shell
bash prepare/download_smpl_model.sh
bash prepare/prepare_clip.sh
bash prepare/download_t2m_evaluators.sh
```

#### Pre-trained model

Run the script to download the pre-train models:


```shell
bash prepare/download_pretrained_models.sh
```

#### Dataset
Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup. Copy the result dataset to our repository:
```shell
cp -r ../HumanML3D/HumanML3D ./datasets/humanml3d
```

## Training
The training setup can be adjusted in the config files: `*.yaml` in /configs.

#### Traning Stage 1 (VAE+SAN):

Run the following script:
```shell
python train.py --cfg ./configs/config_stage1.yaml --cfg_assets ./configs/assets.yaml --nodebug
```

#### Training Stage 2(Conditional motion latent diffusion):

Run the following script:
```shell
python train.py  --cfg ./configs/config_stage2.yaml  --cfg_assets ./configs/assets.yaml  --nodebug
```

## Evaluation
Please first put the tained model checkpoint path to `TEST.CHECKPOINT` in `config_stage1.yaml` and `config_stage2.yaml`.

#### Stage 1:
To evaluate reconstruction performance of stage 1 model, run the following command:
```shell
python test.py --cfg ./configs/config_stage1.yaml --cfg_assets ./configs/assets.yaml 
```

#### Stage 2:
To evaluate motion generation performance of stage 1 model, run the following command:
```shell
python test.py --cfg ./configs/config_stage2.yaml --cfg_assets ./configs/assets.yaml 
```


## Visualizing generated samples

We support text file (for text-to-motion) and npy file(for control signal on motion editing) as input.
The generated/edited motions are npy files.

### Text-to-Motion
```shell
python visualize_test.py --cfg ./configs/config_stage2.yaml  --cfg_assets ./configs/assets.yaml --example ./demo/example.txt
```


### Motion editing
```shell
python visualize_test.py --cfg ./configs/config_stage2.yaml  --cfg_assets ./configs/assets.yaml --example ./demo/example.txt --editing --control ./demo/control_example_start_end.npy
```



## Citation
```bibtex
@article{uchida2024mola,
        title={MoLA: Motion Generation and Editing with Latent Diffusion Enhanced by Adversarial Training},
        author={Uchida, Kengo and Shibuya, Takashi and Takida, Yuhta and Murata, Naoki and Takahashi, Shusuke and Mitsufuji, Yuki},
        journal={arXiv preprint arXiv:2406.01867},
        year={2024}
      }
```

## Reference
Part of the code is borrowed from the following repositories. 
We would like to thank the authors of these repos for their excellent work: 
[MLD](https://github.com/ChenFengYe/motion-latent-diffusion),
[HumanML3D](https://github.com/EricGuo5513/HumanML3D),
[MPGD](https://github.com/KellyYutongHe/mpgd_pytorch/),
[SAN](https://github.com/sony/san),
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT).


## License
This code is distributed under a [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, SMPL-X, and uses datasets which each have their own respective licenses that must also be followed.