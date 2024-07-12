# This code is based on https://github.com/ChenFengYe/motion-latent-diffusion under the MIT license.

import numpy as np
import torch
import random


def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SEED = 10
EVALSEED = 0
# Provoc warning: not fully functionnal yet
# torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

fixseed(SEED)
