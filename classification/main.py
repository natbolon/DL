#!/usr/bin/env python3
""" File to solve the first miniproject which is classification """

import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np

import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
import dlc_practical_prologue as prologue


__author__ = 'Eugène Lemaitre, Natalie Bolón Brun, Louis Munier'
__version__ = '0.1'


class DataStored:
    """A class to well store data to have a cleaner code."""
    def __init__(self, input, classes, target):
        self.input = input
        self.classes = classes
        self.target = target


class DataStoredValid(DataStored):
    """"A class to well store data with validation set to have a cleaner code."""
    def __init__(self, input, classes, target, valid_input, valid_classes, valid_target):
        DataStored.__init__(self, input, classes, target)
        self.valid_input = valid_input
        self.valid_classes = valid_classes
        self.valid_target = valid_target


def import_data(N, normalize):
    """Function to import dataset from prologue"""
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)

    # Normalize data
    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_classes, train_target, test_input, test_classes, test_target


def to_one_hot(tensor):
    one_hot = torch.zeros((tensor.size(0), 10)).type(torch.FloatTensor)
    one_hot[list(range(0,tensor.size(0))), tensor] = 1
    return one_hot


def split_data(train_input, train_classes, test_input, test_classes):
    """Split data into two set, pictures up/down."""
    train_input_up = Variable(
        train_input[:, 0, :, :].reshape(train_input.size(0), 1, train_input.size(2), train_input.size(3)))
    train_input_down = Variable(
        train_input[:, 1, :, :].reshape(train_input.size(0), 1, train_input.size(2), train_input.size(3)))

    train_classes_up = Variable(to_one_hot(train_classes[:, 0]))
    train_classes_down = Variable(to_one_hot(train_classes[:, 1]))

    test_input_up = Variable(
        test_input[:, 0, :, :].reshape(test_input.size(0), 1, test_input.size(2), test_input.size(3)))
    test_input_down = Variable(
        test_input[:, 1, :, :].reshape(test_input.size(0), 1, test_input.size(2), test_input.size(3)))

    test_classes_up = Variable(to_one_hot(test_classes[:, 0]))
    test_classes_down = Variable(to_one_hot(test_classes[:, 1]))

    dict_up = {'train_input': train_input_up, 'train_classes': train_classes_up, 'test_input': test_input_up,
               'test_classes': test_classes_up}
    dict_down = {'train_input': train_input_down, 'train_classes': train_classes_down, 'test_input': test_input_down,
                 'test_classes': test_classes_down}

    return dict_up, dict_down


def validation_set(dict_in_up, dict_in_down, train_target, size):
    rnd = []
    other = np.arange(dict_in_up['train_input'].size(0))

    for i in range(size):
        rnd.append(random.randint(0, dict_in_up['train_input'].size(0) - 1))
        np.delete(other, rnd[-1])

    dict_out_up = {'train_input': dict_in_up['train_input'][other, :, :, :],
                   'train_input_valid': dict_in_up['train_input'][rnd, :, :, :], \
                   'train_classes': dict_in_up['train_classes'][other, :],
                   'train_classes_valid': dict_in_up['train_classes'][rnd, :], \
                   'test_input': dict_in_up['test_input'], 'test_classes': dict_in_up['test_classes']}

    dict_out_down = {'train_input': dict_in_down['train_input'][other, :, :, :],
                     'train_input_valid': dict_in_down['train_input'][rnd, :, :, :], \
                     'train_classes': dict_in_down['train_classes'][other, :],
                     'train_classes_valid': dict_in_down['train_classes'][rnd, :], \
                     'test_input': dict_in_down['test_input'], 'test_classes': dict_in_down['test_classes']}

    return dict_out_up, dict_out_down, train_target[other], train_target[rnd]


def define_device(model, criterion, train_in, train_targ, test_in, test_targ):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    model.to(device)
    criterion.to(device)
    train_in, train_targ = train_in.to(device), train_targ.to(device)
    test_in, test_targ = test_in.to(device), test_targ.to(device)

    return train_in, train_targ, test_in, test_targ


if __name__ == "__main__":
    # Define some variables
    N = 100
    normalize = True

    train_input, train_target, test_input, test_target = import_data(N, normalize)
    train_input, train_target, test_input, test_target = \
        define_device(model, criterion, train_input, train_target, test_input, test_target)

