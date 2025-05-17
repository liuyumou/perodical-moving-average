from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# functions

def exists(val):
    return val is not None


class Lion_PMA(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        accumulate_steps=1, norm_decay=0.01, variance_decay=0.1
        # use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            accumulate_steps=accumulate_steps,
            norm_decay=norm_decay, 
            variance_decay=variance_decay
        )
        self.accumulate_steps = accumulate_steps
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                state['step'] += 1
                step_size = group['lr']
                
                if state['step'] % self.accumulate_steps == 0 and state['step'] != 0:
                    # stepweight decay
                    p.data.mul_(1 - lr * wd)
                    # weight update
                    update = exp_avg.clone().mul_(beta1).add(grad, alpha = (1-beta1)/self.accumulate_steps).sign_()
                    p.add_(update, alpha = -lr)
                    # decay the momentum running average coefficient
                    exp_avg.mul_(beta2).add_(grad, alpha = (1-beta2)/self.accumulate_steps)

                    exp_avg.mul_(self.accumulate_steps)
                else:
                    small_step_times = state['step'] % self.accumulate_steps

                    step_size = step_size / self.accumulate_steps
                    
                    beta1_, beta2_ = group['betas']

                    # weight decay
                    p.data.mul_(1 - lr * wd)
                    # weight update

                    update = exp_avg.clone().mul_((small_step_times) / (small_step_times +1)).add(grad, alpha = (1-beta1_)/(small_step_times+1)).sign_()
                    p.add_(update, alpha = -step_size)
                    # momentum update
                    exp_avg.mul_((small_step_times) / (small_step_times +1)).add_(grad, alpha=(1-beta2_) / (small_step_times+1))
                    

        return loss