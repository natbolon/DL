import random
import torch

def normalize_data(train_input, test_input):
    """
    Normalize the data based on train mean and std
    Modifies input tensors.
    Args:
        train_input:    tensor size=[nbx2x14x14]
        test_input:     tensor size=[nbx2x14x14]
    Returns:
        -

    """
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)


def to_one_hot(tensor):
    """
    Generates vector in one hot coding
    Args:
        tensor: tensor of class values (int from 0 to 9) size=[nb]
    Returns:
        tensor of size              size=[nbx10]
    """
    one_hot = torch.zeros((tensor.size(0), 10)).type(torch.FloatTensor)
    one_hot[list(range(0, tensor.size(0))), tensor[:, 0]] = 1
    return one_hot


def shuffle(t_input, classes, target):
    """
    Shuffle data randomly maintaining the relation between input, classes and target
    Args:
        t_input:    tensor of size=[nbx2x14x14]
        classes:    tensor of size=[nbx10] (already in one-hot format)
        target:     tensor of size=[nb] or size=[nbx2] if converted binary
    Returns:
        shuffled t_input, classes, target

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
    Args:
        target:     tensor of size=[nb] and values [0,1]
    Returns:
        target_bin:  tensor of size=[nbx2]
    """
    target_bin = torch.zeros((target.size(0), 2))
    target_bin[list(range(target.size(0))), target[:]] = 1
    return target_bin


