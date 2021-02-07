import torch
from torch.optim.lr_scheduler import MultiStepLR
from collections import Counter


class WarmUpMultiStepLR(MultiStepLR):
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 warm_up=None):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.iter = 0

        if isinstance(warm_up, dict):
            assert "warmup_iter" in warm_up
            assert "warmup_ratio" in warm_up
            self.warmup_ratio = warm_up['warmup_ratio']
            self.warmup_iter = warm_up['warmup_iter']

        super(WarmUpMultiStepLR, self).__init__(optimizer,
                                                milestones,
                                                gamma=0.1,
                                                last_epoch=-1)



    def get_lr(self, stride=1):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]

    def get_warup_lr(self, cur_iter):
        return [base_lr * self.warmup_ratio + base_lr * \
                (cur_iter / self.warmup_iter) * (1 - self.warmup_ratio)
                for base_lr in self.base_lrs]

    def step_iter(self, cur_iter):
        if cur_iter < self.warmup_iter:
            values = self.get_warup_lr(cur_iter)
        else:
            values = self.get_lr(cur_iter)

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


    def step_epoch(self):
        self.last_epoch += 1




