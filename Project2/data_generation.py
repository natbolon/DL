import matplotlib.pyplot as plt

import math

import torch
from torch import empty

def generate_disc_set(nb):
    """
    Uniform sampling the the space [0,1]x[0,1] and assign labels according to the region location
    :param nb: Number of samples to generate
    :return
    X: torch tensor of size [nb, 2] with the coordinates x,y of each point
    Y: torch tensor of size [nb] with binary entries (0,1) according to the label of each sample
    """
    X = empty(nb, 2).uniform_(0, 1)
    Y = empty(X.size())

    Y[:, 0] = ((X - 0.5).norm(dim=1) > math.sqrt(1 / (2 * math.pi))).type(torch.LongTensor)
    Y[:, 1] = ((X - 0.5).norm(dim=1) <= math.sqrt(1 / (2 * math.pi))).type(torch.LongTensor)

    return X, Y


def plot_disc(data_in, data_target, title):
    """
    Plot samples
    :param data_in: tensor of size [nb, 2] with coordinates x,y of each sample
    :param data_target: tensor of size [nb] with label of each sample
    :param title: (str) title for the plot

    """
    fs = 20
    fig = plt.figure(figsize=(6,6))
    plt.scatter(data_in[(data_target[:, 1] == 1), 0], data_in[(data_target[:, 1] == 1), 1], color="c", s=20)
    plt.scatter(data_in[(data_target[:, 1] == 0), 0], data_in[(data_target[:, 1] == 0), 1], color="g", s=20)
    plt.title(title, fontsize=fs)
    plt.legend(["1", "0"])
    plt.savefig('output/{}.pdf'.format(title))
    plt.close(fig)


def plot_result(data_in, data_target, data_class, train=True, fname=None):
    """
    Plot samples with labels produced from the model
    :param data_in: tensor of size [nb, 2] with coordinates x,y of each sample
    :param data_target: tensor of size [nb] with label of each sample
    :param data_class: tensor of size [nb] with label predicted by the model
    :param train: (boolean) for title and file name to distinguish train and test graphics
    :param fname: (str). If None, do not save image. Otherwise, save image in output folder with name fname.

    """
    fs = 20
    one_id_as_one = torch.mul((data_target[:, 1] == 1), (data_class == 1))
    one_id_as_zero = torch.mul((data_target[:, 1] == 1), (data_class == 0))
    zero_id_as_one = torch.mul((data_target[:, 1] == 0), (data_class == 1))
    zero_id_as_zero = torch.mul((data_target[:, 1] == 0), (data_class == 0))

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(data_in[one_id_as_one, 0], data_in[one_id_as_one, 1], color="c", s=20)
    plt.scatter(data_in[zero_id_as_zero, 0], data_in[zero_id_as_zero, 1], color="g", s=20)
    plt.scatter(data_in[one_id_as_zero, 0], data_in[one_id_as_zero, 1], color="r", s=20)
    plt.scatter(data_in[zero_id_as_one, 0], data_in[zero_id_as_one, 1], color="y", s=20)

    if train:
        plt.title("Result on train data", fontsize=fs)
        f_n = '_train_errors'

    else:
        plt.title("Result on test data", fontsize=fs)
        f_n = '_test_errors'

    plt.legend(["1 id as 1", "0 id as 0", "1 id as 0", "0 id as 1"])

    if isinstance(fname, str):
        fname = fname + f_n
        plt.savefig('output/{}.pdf'.format(fname), bbox_inches='tight')

    plt.close(fig)


def plot_loss(epochs, loss, fname=None):
    """
    Generate plot of loss curve
    :param epochs: list of range(0, n_epochs)
    :param loss: list of values of the loss along the epochs
    :param fname: (str). If None, do not save image. Otherwise, save image in output folder with name fname.
    """
    fs = 20
    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, loss)
    plt.title("Loss", fontsize=fs)
    plt.xlabel("Epoch", fontsize=fs)
    plt.ylabel("Loss", fontsize=fs)
    if isinstance(fname, str):
        plt.savefig('output/{}_loss.pdf'.format(fname), bbox_inches='tight')
    plt.close(fig)


