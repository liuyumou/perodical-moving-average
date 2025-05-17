import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import numpy as np


class AdamW_PMA(Optimizer):
    """This is a self-defined optimizer based on AdamW.
        the step size is based on the consine similarity between the gradient and the momentum.
        the momentum is the exponential moving average of the gradient.
    """
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, accumulate_steps=1, weight_decay=0.01, norm_decay=0.01, variance_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, accumulate_steps=accumulate_steps, weight_decay=weight_decay, norm_decay=norm_decay, variance_decay=variance_decay)
        self.accumulate_steps = accumulate_steps
        super(AdamW_PMA, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step_size = group['lr']
                large_step_times = state['step'] // self.accumulate_steps + 1
                if state['step'] % self.accumulate_steps == 0 and state['step'] != 0:
                    exp_avg.mul_(beta1).add_(grad, alpha=(1-beta1)/self.accumulate_steps)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2)/self.accumulate_steps)

                    # debias
                    exp_avg = exp_avg / (1 - beta1 ** large_step_times)
                    exp_avg_sq = exp_avg_sq / (1 - beta2 ** large_step_times)

                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    p.data.mul_(1 - step_size * group['weight_decay'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    exp_avg.mul_(self.accumulate_steps)
                    exp_avg_sq.mul_(self.accumulate_steps)

                else:
                    small_step_times = state['step'] % self.accumulate_steps
                    step_size = step_size / np.sqrt(self.accumulate_steps)
                    
                    beta1_, beta2_ = group['betas']

                    exp_avg.mul_((small_step_times) / (small_step_times +1)).add_(grad, alpha=(1-beta1_) / (small_step_times+1))
                    exp_avg_sq.mul_((small_step_times) / (small_step_times +1)).addcmul_(grad, grad, value=(1-beta2_) / (small_step_times+1))
                    
                    
                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    
                    p.data.mul_(1 - step_size * group['weight_decay'])
                    p.data.addcdiv_(exp_avg/(1-beta1**large_step_times), denom/(1-beta2**large_step_times), value=-step_size)
        return loss