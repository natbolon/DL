import torch
from torch import empty

from Module import Module
from Module import Module
from Linear import Linear
from Activation import Tanh, Relu, Sigmoid
from Loss import LossMSE, CrossEntropy
from Sequential import Sequential

class Optimizers():
    def __init__(self):
        self.s = torch.empty((0,0))
        
    def __call__(self):
        raise NotImplementedError('Optimizers : __call__ function is not implemented')

    def param(self):
        return []


class Sgd(Optimizers):
    def __init__(self):
        Optimizers.__init__(self)

    def __call__(self, *input, eta=0.1):
        for l in input[0]:
            if isinstance(l, Linear):
                l.bias -= eta * l.gradwrtbias
                l.weight -= eta * l.gradwrtweight

    def param(self):
        return []
        


class DecreaseSGD(Optimizers):
    def __init__(self):
        Optimizers.__init__(self)

    def __call__(self, epoch, *input, eta=0.1, beta=0.1):
        for l in input[0]:
            if isinstance(l, Linear):
                l.bias -= eta/((1+beta*epoch)) * l.gradwrtbias
                l.weight -= eta/((1+beta*epoch)) * l.gradwrtweight


    def param(self):
        return []


