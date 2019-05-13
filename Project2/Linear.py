import math
import torch
from torch import empty

from Module import Module



class Linear(Module):
    def __init__(self, dim_in, dim_out, dropout=None):
        self.x = torch.empty((0,0))
        self.s = torch.empty((0,0))
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.bias = torch.empty(1, dim_out).normal_(1, 0.5) #initialize by default with normal distribution (1,0.5)
        self.weight = torch.empty(dim_out, dim_in).normal_(1, 0.5)
            
        self.gradwrtbias = torch.empty((0,0))
        self.gradwrtweight = torch.empty((0,0))
        
        if dropout:
            if dropout > 1 or dropout < 0:
                raise ValueError('Linear : Dropout must be a percentage value (comprise between 0 and 1)')
            else:
                self.dropout = dropout*100
        else:
            self.dropout = dropout
                
        self.dropout_mask = torch.empty((0,0))
    
    def forward(self, *input):
        self.x = input[0]
        self.s = self.x.mm(self.weight.t()) + self.bias
        
        if self.dropout :
            self.update_dropout()
            return self.s * self.dropout_mask
        else :
            return self.s

    def backward(self, *gradwrtoutput):
        self.gradwrtbias = torch.ones(1, self.x.size(dim=0)).mm(gradwrtoutput[0])
        self.gradwrtweight = gradwrtoutput[0].t().mm(self.x)
        return gradwrtoutput[0].mm(self.weight)
        
    def param(self):
        return [self.bias, self.weight]
        
    def define_parameters(self, weight, bias):
        if weight.size() != (self.dim_out, self.dim_in):
            raise ValueError('Linear : weight size must match ({}, {})'.format(self.dim_out, self.dim_in))
        else:
            self.weight = weight
            
        if bias.size() != (1, self.dim_out):
            raise ValueError('Linear : bias size must match ({}, {})'.format(1, self.dim_out))
        else:
            self.bias = bias
        
    def normalize_parameters(self, mean, std):
        self.bias = self.bias.normal_(mean=mean, std=std)
        self.weight = self.weight.normal_(mean=mean, std=std)
        
    def uniform_parameters(self):
        self.bias = self.bias.uniform_()
        self.weight = self.weight.uniform_()
        
    def xavier_parameters(self):
        std = math.sqrt(2 / (self.weight.size(0) + self.weight.size(1)))
        self.normalize_parameters(0, std)
        
    def update_parameters(self, eta):
        self.bias -= eta * self.gradwrtbias
        self.weight -= eta * self.gradwrtweight
        
    def update_dropout(self):
        self.dropout_mask = torch.randint(101, self.s.size())
        self.dropout_mask = (self.dropout_mask >= self.dropout).type(torch.FloatTensor)
