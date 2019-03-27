#!/usr/bin/env python3
# File to solve the first miniproject which is classification

# lm270319.1058

import torch
import dlc_practical_prologue as prologue


def import_data(N, normalize):
    train_input, train_target, train_classes, \
    test_input, test_target, test_classes = prologue.generate_pair_sets(N)

    # Normalize data
    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target


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

