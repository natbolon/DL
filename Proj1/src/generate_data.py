import random
import torch

def normalize_data(train_input, test_input):
    """
    Normalize the data based on train mean and std
    Modifies input tensors.
    :param train_input: tensor size=[nbx2x14x14]
    :param test_input: tensor size=[nbx2x14x14]
    """

    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)


def to_one_hot(tensor):
    """
    Generates vector in one hot coding
    :param tensor: tensor of class values (int from 0 to 9) size=[nb]
    :return: size=[nb,10]
    """

    one_hot = torch.zeros((tensor.size(0), 10)).type(torch.FloatTensor)
    one_hot[list(range(0, tensor.size(0))), tensor[:, 0]] = 1
    return one_hot


def shuffle(t_input, classes, target):
    """
    Shuffle data randomly maintaining the relation between input, classes and target
    :param t_input: tensor of size=[nbx2x14x14]
    :param classes: tensor of size=[nb,10] (if already in one-hot format) or [nb,1] if integer class
    :param target: tensor of size=[nb] or size=[nbx2] if converted binary
    :return:
    """

    idx = [i for i in range(t_input.size(0))]
    random.shuffle(idx)
    if len(target.shape) == 1:
        return t_input[idx, :, :, :], classes[idx, :], target[idx]
    else:
        return t_input[idx, :, :, :], classes[idx, :], target[idx,:]


def binarize(target):
    """
    Binarize target tensor
    :param target: tensor of size=[nb] and values [0,1]
    :return: tensor of size=[nb,2]
    """

    target_bin = torch.zeros((target.size(0), 2))
    target_bin[list(range(target.size(0))), target[:]] = 1
    return target_bin