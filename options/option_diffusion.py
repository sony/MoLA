import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    
    ## optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-epochs', default=10, type=int, help='number of total epochs for warmup')
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vae arch
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vae-act', type=str, default='leakyrelu', choices = ['relu', 'silu', 'gelu', 'leakyrelu'], help='dataset directory')
    parser.add_argument("--latent-dim", type=int, default=16, help="size of latent space")
    parser.add_argument('--encoder-input', type=str, default='root_pos_rot', choices = ['all', 'root_pos_rot'], help='How to input motion feature into Encoder')
    parser.add_argument('--decoder-output', type=str, default='all', choices = ['all', 'except_rot'], help='How to output motion feature on Decoder')

    ## diffusion arch
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-depth", type=int, default=9, help="depth of transformer layers")
    parser.add_argument("--num-head", type=int, default=4, help="head of transformer layers")

    parser.add_argument("--inference-timestep", type=int, default=50, help="number of inference timestep")
    parser.add_argument("--cfg-guidance-scale", type=float, default=11, help="number of inference timestep")

    ## resume
    parser.add_argument("--resume-vae", type=str, default=None, help='resume pth for VAE')
    parser.add_argument("--resume-dit", type=str, default=None, help='resume pth for DiT')
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vae-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out-dir')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument("--edit-mode", type=str, default=None, choices = ['inbetweening', 'upper_edit', 'path'],help='type of motion editing')
    parser.add_argument("--edit-scale", type=float, default=0.6, help='stepsize for mggd')
    parser.add_argument("--prompt", type=str, default=None,help='text prompt')
    
    
    
    return parser.parse_args()