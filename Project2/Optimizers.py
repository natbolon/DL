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
    def __init__(self, eta=0.1):
        """
        :param eta: learning rate
        """
        Optimizers.__init__(self)
        self.eta=eta

    def __call__(self, *input):
        # Iterate through the layers of the network and update parameters of the Linear ones
        for l in input[0]:
            if isinstance(l, Linear):
                # Update layer parameters
                l.bias -= self.eta * l.gradwrtbias
                l.weight -= self.eta * l.gradwrtweight

    def param(self):
        return []
        


class DecreaseSGD(Optimizers):
    def __init__(self, eta):
        """
        :param eta: learning rate
        """
        Optimizers.__init__(self)
        self.eta = eta

    def __call__(self, epoch, *input, beta=0.1):
        # Iterate through the layers of the network and update parameters of the Linear ones
        for l in input[0]:
            if isinstance(l, Linear):
                # Update layer parameters
                l.bias -= self.eta/((1+beta*epoch)) * l.gradwrtbias
                l.weight -= self.eta/((1+beta*epoch)) * l.gradwrtweight


    def param(self):
        return []
    
    
    

class Adam(Optimizers):
    def __init__(self, eta, beta1=0.9, beta2=0.99, delta=0.1):
        """

        :param eta: learning rate
        :param beta1: first order decaying parameter
        :param beta2: second order decaying parameter
        :param delta: damping coefficient
        """
        Optimizers.__init__(self)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.delta = delta
        self.m1_w = None
        self.m2_w = None
        

    def __call__(self, *input):
        # Initialize moment estimators
        if self.m1_w is None:
            self.m1_w = [torch.zeros(l.gradwrtweight.size()) for l in input[0] if isinstance(l, Linear)]
            
        if self.m2_w is None:
            self.m2_w = [torch.zeros(l.gradwrtweight.size()) for l in input[0] if isinstance(l, Linear)]  

        # Iterate through the layers of the network and update parameters of the Linear ones
        i = 0
        for l in input[0]:
            if isinstance(l, Linear):
                # Compute first and second order moment estimators
                self.m1_w[i] = self.beta1 * self.m1_w[i] + (1 - self.beta1) * l.gradwrtweight
                self.m2_w[i] = self.beta2 * self.m2_w[i] + (1 - self.beta2) * l.gradwrtweight * l.gradwrtweight

                # Correct bias on moment estimators
                m1_w = self.m1_w[i] / (1 - self.beta1 ** (i + 1))
                m2_w = self.m2_w[i] / (1 - self.beta2 ** (i + 1))

                # Update layer parameters
                l.weight -= self.eta * m1_w / (self.delta + torch.sqrt(m2_w))
                l.bias -= self.eta*l.gradwrtbias
                i += 1

               

