import torch
from torch import empty

from Module import Module


class Loss(Module):
    def __init__(self):
        self.output = empty((0,0))
        self.target_output = empty((0,0))
    
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    
class LossMSE(Loss):   
    def __init__(self):
        Loss.__init__(self)
    
    def forward( self , output, target_output ):
        self.output = output
        self.target_output = target_output
        loss = (output - target_output).pow(2).mean()  # (dim=0).sum()
        return loss


    def backward ( self ):
        return 2*(self.output - self.target_output)/self.output.numel()  # size(dim=0)

    def param ( self ) :
        return []
    
class CrossEntropy(Loss):
    def __init__(self):
        Loss.__init__(self)
    
    def forward( self , output, target_output ):
        self.output = output
        self.target_output = target_output
        loss =  -1./output.size(dim=0) * ( output[ (torch.arange(0, output.size(dim=0) )).type(torch.long), target_output ].sum() - output.exp().sum(dim=1).log().sum() ) 
        return loss
        
    def backward(self):
        grad = empty(self.output.size())
        for i in range(0, self.output.size(dim=0)):
            grad[i,:] = self.output[i,:].exp().div_( self.output[i,:].exp().sum())
       
        grad[(torch.arange(0, self.output.size(dim=0))).type(torch.long), self.target_output] -= 1
        grad.div_(self.output.size(dim=0))
        return grad
            

    def param(self):
        return []


