import torch

from Module import Module


class Activation(Module):
    def __init__(self):
        self.s = torch.empty((0,0))

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

class Tanh(Activation):
    def __init__(self):
        Activation.__init__(self)
        
    def forward(self, x):
        """
        :param x: tensor generated as output of a linear layer
        """
        self.s = x
        return x.tanh()

    def backward(self, *gradwrtoutput):
        """
        Computes gradient wrt to the function
        :param gradwrtoutput: gradient accumulated from the posterior layers
        :return: updated gradient
        """
        return (1 - (self.s.tanh()).pow(2)) * (gradwrtoutput[0])
        
    def param(self):
        return []
    
    
class Relu(Activation):
    def __init__(self):
        Activation.__init__(self)

    
    def forward(self, x):
        """
        :param x: tensor generated as output of a linear layer
        """
        self.s = x
        y = torch.empty(self.s.size()).zero_()
        return torch.max(x, y)
    
    def backward(self, *gradwrtoutput):
        """
        Computes gradient wrt to the function
        :param gradwrtoutput: gradient accumulated from the posterior layers
        :return: updated gradient
        """
        y = torch.empty(gradwrtoutput[0].size()).zero_()
        return (torch.eq(y, torch.min(y, self.s))).type(torch.FloatTensor).mul(gradwrtoutput[0])

    def param(self):
        return []


class Sigmoid(Activation):
    def __init__(self, p_lambda):
        Activation.__init__(self)
        self.p_lambda = p_lambda
    
    def forward(self, x):
        """
        :param x: tensor generated as output of a linear layer
        """
        self.s = x
        return 1/(1 + torch.exp(-self.p_lambda*x))

    def backward(self, *gradwrtoutput):
        """
        Computes gradient wrt to the function
        :param gradwrtoutput: gradient accumulated from the posterior layers
        :return: updated gradient
        """
        return torch.exp(-self.s * self.p_lambda) / (torch.exp(-self.s * self.p_lambda) + 1).pow(2) * (gradwrtoutput[0])


    def param(self):        
        return []
