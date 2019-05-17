import math
import torch

from Module import Module



class Linear(Module):
    """ Child class of Module. Implements a fully connected layer"""
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
        """
        Perform forward pass
        :param x: input tensor if first layer or output given by a previous layer
        """
        # Compute forward pass and store input
        self.x = input[0]
        self.s = self.x.mm(self.weight.t()) + self.bias
        
        if self.dropout :
            self.update_dropout()
            return self.s * self.dropout_mask
        else :
            return self.s

    def backward(self, *gradwrtoutput):
        """
        Perform backward pass to compute gradient
        """
        # Compute backward pass and store gradient
        self.gradwrtbias = torch.ones(1, self.x.size(dim=0)).mm(gradwrtoutput[0])
        self.gradwrtweight = gradwrtoutput[0].t().mm(self.x)
        return gradwrtoutput[0].mm(self.weight)
        
    def param(self):
        return [self.bias, self.weight]
        
    def define_parameters(self, weight, bias):
        """
        Initialize weights and bias if previously given

        """
        if weight.size() != (self.dim_out, self.dim_in):
            raise ValueError('Linear : weight size must match ({}, {})'.format(self.dim_out, self.dim_in))
        else:
            self.weight = weight
            
        if bias.size() != (1, self.dim_out):
            raise ValueError('Linear : bias size must match ({}, {})'.format(1, self.dim_out))
        else:
            self.bias = bias
        
    def normalize_parameters(self, mean, std):
        """
        Initialize parameters with normal distribution
        :param mean: mean of the normal distribution
        :param std: standard deviation of the normal distribution
        """
        # Initialization of weight and bias with normally distributed values
        self.bias = self.bias.normal_(mean=mean, std=std)
        self.weight = self.weight.normal_(mean=mean, std=std)
        
    def uniform_parameters(self):
        """
        Initialize parameters with uniform distribution between 0 and 1
        """
        # Initialization of weight and bias with uniformly distributed values between [0,1]
        self.bias = self.bias.uniform_()
        self.weight = self.weight.uniform_()
        
    def xavier_parameters(self):
        """
        Initialize parameters with uniform distribution between 0 and 1
        """
        # Initialization of parameters with normally distributed values with std= sqrt(2/(width_layer + height_layer))
        std = math.sqrt(2 / (self.weight.size(0) + self.weight.size(1)))
        self.normalize_parameters(0, std)
        
    def update_parameters(self, eta):
        """
        Perform parameters update by gradient descent
        :param eta: learning rate
        """
        # Update parameters by gradient descent if no optimizer is specified
        self.bias -= eta * self.gradwrtbias
        self.weight -= eta * self.gradwrtweight
        
    def update_dropout(self):
        """
        Update dropout mask
        """
        # Update dropout mask
        self.dropout_mask = torch.randint(101, self.s.size())
        self.dropout_mask = (self.dropout_mask >= self.dropout).type(torch.FloatTensor)
