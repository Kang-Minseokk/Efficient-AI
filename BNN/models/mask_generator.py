import torch
from torch.special import erf
import numpy as np
from abc import ABCMeta, abstractmethod
from models.temp_scheduler import *

import pdb

ERFINV = {0.5: 0.4769362762044698733814, 
         -0.5: -0.4769362762044698733814}
    
class MaskGenerator(metaclass=ABCMeta):
    def __init__(self, mask_type, scheduler_type, temp_init, temp_min, T_max):
        self.mask_type = mask_type
        self.scheduler_type = scheduler_type
        self._set_scheduler(self.scheduler_type, temp_init, temp_min, T_max)

    def _set_scheduler(self, scheduler_type, temp_init=None, temp_min=None, T_max=None):
        if self.scheduler_type == "const":
            self.scheduler = ConstantScheduler(temp_init, temp_min, T_max)
        elif self.scheduler_type == "cosine":
            self.scheduler = CosineScheduler(temp_init, temp_min, T_max)
        elif self.scheduler_type == "gaussian":
            self.scheduler = GaussianScheduler(temp_init, temp_min, T_max)
        else:
            raise NotImplementedError(f"SCHEDULER ERROR: [{self.scheduler_type}] is not available.")
    
    @abstractmethod
    def _get_prob(self, grad, b_weight, temp):
        pass

    def get_mask(self, grad, b_weight):
        temp = self.scheduler.get_temp(update=True, update_step=(grad.std() if self.scheduler_type=="gaussian" else None))
        prob = self._get_prob(grad, b_weight, temp)
        assert ((prob>=0)&(prob<=1)).all().item(), f"RUNTIME ERROR: something is wrong in _get_prob()."
        mask = torch.rand(grad.shape, device=grad.device) < prob
        return mask

class EMPGenerator(MaskGenerator):
    def _get_prob(self, grad, b_weight, temp):
        prob = torch.zeros_like(grad, device=grad.device)
        prob_pos = erf(temp*grad) - erf(torch.min(torch.zeros_like(grad, device=grad.device), temp*grad))
        prob_neg = erf(torch.max(torch.zeros_like(grad, device=grad.device), temp*grad)) - erf(temp*grad)
        prob[b_weight>0] = prob_pos[b_weight>0]
        prob[b_weight<0] = prob_neg[b_weight<0]
        return prob

class MMPGenerator(MaskGenerator):
    def _get_prob(self, grad, b_weight, temp):
        prob = torch.zeros_like(grad, device=grad.device)
        prob[(b_weight>0)&(grad>=ERFINV[0.5]/temp)] = 1
        prob[(b_weight<0)&(grad<=ERFINV[-0.5]/temp)] = 1
        prob[(b_weight>0)&(grad<ERFINV[0.5]/temp)] = 0
        prob[(b_weight<0)&(grad>ERFINV[-0.5]/temp)] = 0
        return prob

class RANDGenerator(MaskGenerator):
    def _get_prob(self, grad, b_weight, temp):
        prob = torch.ones_like(grad, device=grad.device)
        prob *= temp
        prob[prob>1] = 1
        prob[prob<0] = 0
        return prob

def get_mask_generator(mask_type, scheduler_type, temp_init, temp_min, T_max):
    if mask_type=="EMP":
        mask_generator = EMPGenerator(mask_type=mask_type, scheduler_type=scheduler_type, temp_init=temp_init, temp_min=temp_min, T_max=T_max)
    elif mask_type=="MMP":
        mask_generator = MMPGenerator(mask_type=mask_type, scheduler_type=scheduler_type, temp_init=temp_init, temp_min=temp_min, T_max=T_max)
    elif mask_type=="RAND":
        mask_generator = RANDGenerator(mask_type=mask_type, scheduler_type=scheduler_type, temp_init=temp_init, temp_min=temp_min, T_max=T_max)
    else:
        raise NotImplementedError(f"MASK ERROR: [{self.mask_type}] is not available.")
    return mask_generator