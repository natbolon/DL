import random
import torch

def normalize_data(train_input, test_input):
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)


def to_one_hot(tensor):
    one_hot = torch.zeros((tensor.size(0), 10)).type(torch.FloatTensor)
    one_hot[list(range(0, tensor.size(0))), tensor[:, 0]] = 1
    return one_hot


def shuffle(t_input, classes, target):
    idx = [i for i in range(t_input.size(0))]
    random.shuffle(idx)
    return t_input[idx, :, :, :], classes[idx, :], target[idx, :]


def binarize(target):
    target_bin = torch.zeros((target.size(0), 2))
    target_bin[list(range(target.size(0))), target[:]] = 1
    return target_bin


