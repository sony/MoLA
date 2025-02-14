import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=20000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=10, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[10000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--kl-weight", type=float, default=0.01, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')
    
    parser.add_argument('--gan-weight', type=float, default=0.01, help='hyper-parameter for the GAN loss')
    

    ## vae arch
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vae-act', type=str, default='leakyrelu', choices = ['relu', 'silu', 'gelu', 'leakyrelu'], help='dataset directory')
    parser.add_argument('--vae-norm', type=str, default=None, help='dataset directory')
    parser.add_argument("--latent-dim", type=int, default=16, help="size of latent space")
    parser.add_argument('--encoder-input', type=str, default='all', choices = ['all', 'root_pos_rot', 'root_pos'], help='How to input motion feature into Encoder')
    parser.add_argument('--decoder-output', type=str, default='all', choices = ['all', 'except_rot'], help='How to output motion feature on Decoder')

    
    
    ## resume
    parser.add_argument("--resume-vae", type=str, default=None, help='resume pth for VAE')
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    
    
    return parser.parse_args()