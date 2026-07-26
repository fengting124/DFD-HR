from torch.optim.lr_scheduler import _LRScheduler


class LinearDecayLR(_LRScheduler):
    """Keep base LR, then linearly decay it to zero by the final epoch.

    This is a compatibility scheduler. Paper-aligned DFD-HR uses no scheduler
    and therefore keeps Adam's learning rate fixed.
    """

    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay = start_decay
        self.n_epoch = n_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch = self.n_epoch
        base_lr = self.base_lrs[0]
        start_decay = self.start_decay
        if last_epoch > start_decay:
            lr = base_lr - base_lr / (n_epoch - start_decay) * (
                last_epoch - start_decay
            )
        else:
            lr = base_lr
        return [lr]
