"""Optimization Utilities (Schedulers)."""

import math
import torch

class WarmupCosineScheduler:
    """Cosine Annealing with Linear Warmup."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        lr_values = self._get_lr_values()
        for param_group, lr in zip(self.optimizer.param_groups, lr_values):
            param_group['lr'] = lr
    
    def _get_lr_values(self):
        if self.step_count <= self.warmup_steps:
            factor = self.step_count / max(1, self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class NoamScheduler:
    """
    Noam learning rate scheduler as described in Attention Is All You Need.
    lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def state_dict(self):
        return {
            'step_num': self.step_num,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
        self.factor = state_dict['factor']
        
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
            
    def _get_lr(self):
        return self.factor * (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Factory for standard lambda scheduler."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
