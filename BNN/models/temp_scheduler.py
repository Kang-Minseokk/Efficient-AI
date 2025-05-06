import torch
import numpy as np
from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):
    def __init__(self, temp_init, temp_min, T_max):
        self.temp_init, self.temp_min = temp_init, temp_min
        self.T_cur, self.T_max = 0, T_max

    @abstractmethod
    def get_temp(self, update=True, update_step=None):
        pass

class ConstantScheduler(Scheduler):
    def get_temp(self, update=True, update_step=None):
        temp = self.temp_init
        if update:
            self.T_cur = min(self.T_cur+1, self.T_max)
        return temp

class CosineScheduler(Scheduler):
    def get_temp(self, update=True, update_step=None):
        temp = self.temp_min + (self.temp_init-self.temp_min)*(1+np.cos(np.pi*self.T_cur/self.T_max))/2
        if update:
            self.T_cur = min(self.T_cur+1, self.T_max)
        return temp

class GaussianScheduler(Scheduler):
    def get_temp(self, update=True, update_step=None):
        temp = self.temp_init
        if update:
            self.temp_init = (self.temp_init/(1+2*(self.temp_init*update_step)**2)).item()
            self.T_cur = min(self.T_cur+1, self.T_max)
        return temp