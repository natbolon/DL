import torch
from torch import empty

from Module import Module
from Linear import Linear
from Activation import Activation, Tanh, Relu, Sigmoid
from Loss import Loss, LossMSE, CrossEntropy


class Sequential(Module):
    """Child class of Module, create a NN by giving a sequence of layers"""
    
    def __init__(self, *sequence, loss=None):
        self.sequence = []
        
        if not loss:
            raise ValueError('Wrong argument given to Sequential a loss function must be given.')
        else:
            self.loss = loss
        
        for i, arg in enumerate(sequence):
            if isinstance(arg, Linear) and i%2 == 0:
                self.sequence.append(arg)
            elif isinstance(arg, Activation) and i%2 == 1:
                self.sequence.append(arg)
            else:
                if i%2 == 0:
                    raise ValueError('Wrong argument given to Sequential. A Linear layer was expected as argument {}.'.format(i+1))
                elif isinstance(arg, Loss) and i != len(sequence):
                    raise ValueError('Wrong argument given to Sequential. The loss must be the last argument given.')
                else:
                    raise ValueError('Wrong argument given to Sequential an Activation function was expected as argument {}.'.format(i+1))
                    
    def forward(self, x):
        for step in self.sequence:
            x = step.forward(x)
        return x

    def backward(self):
        grad = self.loss.backward()
        for step in reversed(self.sequence):
            grad = step.backward(grad)
    
    def param(self):
        return []
    
    def compute_loss(self, output, target_output):
        if type(self.loss) == CrossEntropy:
            return self.loss.forward(output, target_output.argmax(dim=1))
        else:
            return self.loss.forward(output, target_output)
    
    def normalize_parameters(self, mean, std):
        for step in reversed(self.sequence):
            if step.param() != []:
                step.normalize_parameters(mean, std)
                
    def uniform_parameters(self):
        for step in reversed(self.sequence):
            if step.param() != []:
                step.uniform_parameters()
        
    def xavier_parameters(self):
        for step in reversed(self.sequence):
            if step.param() != []:
                step.xavier_parameters()
                
    def update_parameters(self, eta):
        for step in reversed(self.sequence):
            if step.param() != []:
                step.update_parameters(eta)
