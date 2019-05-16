import torch

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
        """
        Perform forward pass
        :param x: input tensor if first layer or output given by a previous layer
        """
        for step in self.sequence:
            x = step.forward(x)
        return x

    def backward(self):
        """
        Perform backward pass to compute gradient
        """
        grad = self.loss.backward()
        for step in reversed(self.sequence):
            grad = step.backward(grad)
    
    def param(self):
        return []
    
    def compute_loss(self, output, target_output):
        """
        Compute loss
        :param output: tensor generated by the network
        :param target_output: tensor of true labels of the input samples
        :return: number of misclassified samples
        """
        if type(self.loss) == CrossEntropy:
            return self.loss.forward(output, target_output.argmax(dim=1))
        else:
            return self.loss.forward(output, target_output)
    
    def normalize_parameters(self, mean, std):
        """
        Initialize parameters with normal distribution
        :param mean: mean of the normal distribution
        :param std: standard deviation of the normal distribution
        """
        for step in reversed(self.sequence):
            if step.param() != []:
                step.normalize_parameters(mean, std)
                
    def uniform_parameters(self):
        """
        Initialize parameters with uniform distribution between 0 and 1
        """
        for step in reversed(self.sequence):
            if step.param() != []:
                step.uniform_parameters()
        
    def xavier_parameters(self):
        """
        Initialize parameters with normal distribution with variance computed as suggested by X. Glorot
        """
        for step in reversed(self.sequence):
            if step.param() != []:
                step.xavier_parameters()
                
    def update_parameters(self, eta):
        """
        Perform parameters update by gradient descent
        :param eta: learning rate
        """
        for step in reversed(self.sequence):
            if step.param() != []:
                step.update_parameters(eta)
