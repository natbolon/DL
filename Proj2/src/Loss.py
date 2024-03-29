import torch

from Module import Module


class Loss(Module):
    def __init__(self):
        self.output = torch.empty((0,0))
        self.target_output = torch.empty((0,0))
    
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    
class LossMSE(Loss):
    """ Child of class Loss. Implements MSE loss"""
    def __init__(self):
        Loss.__init__(self)
    
    def forward( self , output, target_output ):
        """
        :param output: tensor generated from the model given an input
        :param target_output: tensor with ground truth labels of the input
        :return: (float) loss evaluated through MSE criterion
        """
        self.output = output
        self.target_output = target_output
        loss = (output - target_output).pow(2).mean()  # (dim=0).sum()
        return loss


    def backward ( self ):
        """
        :return: (tensor) gradient of loss with respect to the output of the layer
        """
        return 2*(self.output - self.target_output)/self.output.numel()  # size(dim=0)

    def param ( self ) :
        return []
    
class CrossEntropy(Loss):
    """ Child of class Loss. Implements Cross Entropy loss"""
    def __init__(self):
        Loss.__init__(self)
    
    def forward( self , output, target_output ):
        """
        :param output: tensor generated from the model given an input
        :param target_output: tensor with ground truth labels of the input
        :return: (float) loss evaluated through Cross Entropy criterion
        """
        self.output = output
        self.target_output = target_output
        loss =  -1./output.size(dim=0) * ( output[ (torch.arange(0, output.size(dim=0) )).type(torch.long), target_output ].sum() - output.exp().sum(dim=1).log().sum() ) 
        return loss
    
    def backward(self):
        """
        :return: (tensor) gradient of loss with respect to the output of the layer
        """
        grad = self.output.exp().div_(self.output.exp().sum(dim=1).expand(self.output.t().size()).t())
            
        grad[list(range(0, self.output.size(dim=0))), self.target_output] -= 1
        grad.div_(self.output.size(dim=0))
        return grad
            

    def param(self):
        return []
