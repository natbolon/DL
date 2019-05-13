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
    
    
    

class Adam(Optimizers):
    def __init__(self, eta, beta1, beta2, delta):
        Optimizers.__init__(self)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.delta = delta
        self.m1_w = None
        self.m2_w = None
        

    def __call__(self, *input):
        if self.m1_w is None:
            self.m1_w = [torch.zeros(l.gradwrtweight.size()) for l in input[0] if isinstance(l, Linear)]
            
        if self.m2_w is None:
            self.m2_w = [torch.zeros(l.gradwrtweight.size()) for l in input[0] if isinstance(l, Linear)]  
        
        i = 0
        for l in input[0]:
            if isinstance(l, Linear):
                self.m1_w[i] = self.beta1 * self.m1_w[i] + (1 - self.beta1) * l.gradwrtweight
                self.m2_w[i] = self.beta2 * self.m2_w[i] + (1 - self.beta2) * l.gradwrtweight * l.gradwrtweight
                
                m1_w = self.m1_w[i] / (1 - self.beta1 ** (i + 1))
                m2_w = self.m2_w[i] / (1 - self.beta2 ** (i + 1))
                
                l.weight -= self.eta * m1_w / (self.delta + torch.sqrt(m2_w))
                l.bias -= self.eta*l.gradwrtbias
                i += 1

               

