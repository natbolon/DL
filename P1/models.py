from torch import nn
from torch.nn import functional as F



class Net_Conv(nn.Module):
    def __init__(self, nb_hidden):
        super(Net_Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*3*3, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 64*3*3)))
        x = self.fc2(x)
        return x


class Net_Full(nn.Module):
    def __init__(self):
        super(Net_Full, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_fc(nn.Module):
    def __init__(self, n_in):
        super(Net_fc, self).__init__()
        self.fc1 = nn.Linear(n_in, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Net_All(nn.Module):
    def __init__(self, nb_h1, nb_h2, nb_h4, nb_h5):
        super(Net_All, self).__init__()
        self.conv1 = nn.Conv2d(1, nb_h1, kernel_size=5)
        self.conv2 = nn.Conv2d(nb_h1, nb_h2, kernel_size=3)
        self.conv3 = nn.Conv2d(nb_h2, 32, kernel_size=1)
        self.fc1 = nn.Linear(32 * 6 * 6, nb_h4)
        self.fc2 = nn.Linear(nb_h4, 10)
        self.fc3 = nn.Linear(20, nb_h5)
        self.fc4 = nn.Linear(nb_h5, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=1))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, (32 * 6 * 6))))
        x_classes = F.relu(self.fc2(x))
        x_out = F.relu(self.fc3(x_classes.view(-1, 20)))
        x_out = self.fc4(x_out)
        return x_classes, x_out


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
        x_classes = F.relu(self.fc1(x.view(-1, (self.size_h2 * 7 * 7))))
        x_out = F.relu(self.fc2(x_classes.view(-1, 20)))
        x_out = self.fc3(x_out)
        return x_classes, x_out