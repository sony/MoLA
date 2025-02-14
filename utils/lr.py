import torch
import math

class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.00001,
        eta_min: float = 0.00001,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                Optimization method instance
            warmup_epochs (int):
                Number of epochs to perform linear warmup
            max_epochs (int):
                Number of learning epochs used to terminate the cosine curve
            warmup_start_lr (float):
                start learning rate
            eta_min (float):
                lower bound of cosine curve
            last_epoch (int):
                Phase offset of cosine curve
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        return None

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        final_lr (float): The final learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]