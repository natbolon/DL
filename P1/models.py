import torch
from torch import nn
from torch.nn import functional as F



class Net_Conv(nn.Module):
    def __init__(self, nb_hidden, sig=False):
        super(Net_Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 2**5, kernel_size=5)
        self.conv2 = nn.Conv2d(2**5, 2**4, kernel_size=3)
        self.fc1 = nn.Linear((2**4)*6*6, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        self.sig = sig

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=1))
        x = F.relu(self.fc1(x.view(-1, (2**4)*6*6)))
        if self.sig:
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x

# MLP FOR VALUE COMPARISON
class Net_Full(nn.Module):
    def __init__(self):
        super(Net_Full, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #return torch.sigmoid(x)
        return x

# MLP FOR FULL TASK
class Net_fc(nn.Module):
    def __init__(self, n_in):
        super(Net_fc, self).__init__()
        self.fc1 = nn.Linear(n_in, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #return torch.sigmoid(x)
        return x


# NET WITH INTERMEDIATE LOSS - SMALL SIZE
class Net_small_all(nn.Module):
    def __init__(self, nb_h1, nb_h2, nb_h3):
        super(Net_small_all, self).__init__()
        self.size_h2 = nb_h2
        self.conv1 = nn.Conv2d(1, nb_h1, kernel_size=5)
        self.conv2 = nn.Conv2d(nb_h1, nb_h2, kernel_size=3)
        self.fc1 = nn.Linear(nb_h2 * 7 * 7, 10)
        self.fc2 = nn.Linear(20, nb_h3)
        self.fc3 = nn.Linear(nb_h3, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
        x = F.relu(self.conv2(x))
        x_classes = torch.sigmoid(self.fc1(x.view(-1, (self.size_h2 * 7 * 7))))
        x_out = F.relu(self.fc2(x_classes.view(-1, 20)))
        x_out = self.fc3(x_out)
        #return x_classes, torch.sigmoid(x_out)
        return x_classes, x_out

