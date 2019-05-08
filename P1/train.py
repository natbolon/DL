import time
import torch
from torch import nn, optim

from generate_data import shuffle, to_one_hot
from models import Net_Conv, Net_Full, Net_small_all, Net_fc


# Define training function for two-stage model

def train_model(model, train_input, train_target, test_input=0, test_target=None, epochs=25, \
                mini_batch_size=100, lr=1e-3, criterion=None, optimizer=None, verbose=2):
    """
    Training function for single model with single output
    If no test set is provided, the function generates a validation set that is modified each run (cross-validation)

    Args:
        model:          model to be trained
        train_input:    tensor of size [n_samples, 1, 14, 14] or [n_samples, 20]
        train_target:   tensor of size [n_samples] (integer values corresponding to class of the image or target value

        test_input:     tensor of size [n_samples, 1, 14, 14] or [n_samples, 20]
        test_target:    tensor of size [n_samples] (integer values corresponding to class of the image or target value

        verbose: 0 - loss per each epoch
                 1 - loss each 5 epochs
                 2 - modifying loss value along epochs
        other: epochs, mini_batch_size, lr=learning rate, criterion= loss function, optimizer

    Returns:
        loss_store:     storage of the train loss along epochs
        time_store:     storage of the time along epochs
        error_store:    storage of the train error along epochs
        error_store_test:   storage of the test error along epochs

    """

    # use Cross Entropy by default
    if not criterion:
        criterion = nn.CrossEntropyLoss()

    # use SGD by default
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    # initialize list to store loss and error values
    loss_store = []
    error_store = []
    error_store_test = []
    current_time = 0
    time_store = [current_time]

    for e in range(epochs):

        # initialize loss
        sum_loss = 0
        start = time.time()

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.item()
            optimizer.step() # update gradient

        end = time.time()

        error_store.append(compute_nb_errors(model, train_input, train_target))

        # evaluate test error along the iterations
        if not isinstance(test_input, int):
            error_store_test.append(compute_nb_errors(model, test_input, test_target))
        else:
            print('Not test error but train error!')
            error_store_test = error_store

        # store loss and epoch time
        loss_store.append(sum_loss)
        current_time += (end - start)
        time_store.append(current_time)

        if verbose == 0:
            print('Epoch: {}, loss: {:0.5f}'.format(e, sum_loss))
        elif verbose == 1 and e % 5 == 0:
            print('Epoch: {}, loss: {:0.5f}'.format(e, sum_loss))
        elif verbose == 2:
            print("\rTraining: {:0.1f} %, Error: {}".format(100.0 * (e + 1) / epochs, error_store_test[-1]), end="")


    return loss_store, time_store, error_store, error_store_test


#  Define training function for complete model
def train_model_all(model, train_input, train_classes, train_target, test_input=0, test_target=None, epochs=25, \
                    mini_batch_size=100, lr=1e-3, w1=1, w2=1, criterion1=None, criterion2=None, optimizer=None,
                    verbose=2):
    """
        Training function for single model with multiple output.
        If a single loss wants to be used, the corresponding weight must be set to zero
        If no test set is provided, the function generates a validation set that is modified each run (cross-validation)

        Args:
            model:          model to be trained
            train_input:    tensor of size [n_samples, 1, 14, 14] or [n_samples, 20]
            train_target:   tensor of size [n_samples] (integer values corresponding to class of the image or target value

            test_input:     tensor of size [n_samples, 1, 14, 14] or [n_samples, 20]
            test_target:    tensor of size [n_samples] (integer values corresponding to class of the image or target value

            verbose: 0 - loss per each epoch
                     1 - loss each 5 epochs
                     2 - modifying loss value along epochs
            other: epochs, mini_batch_size, lr=learning rate, criterion= loss function, optimizer

        Returns:
            loss_store:     storage of the train loss along epochs
            time_store:     storage of the time along epochs
            error_store:    storage of the train error along epochs
            error_store_test:   storage of the test error along epochs

    """

    # use Cross Entropy by default
    if not criterion1:
        criterion1 = nn.CrossEntropyLoss()
    if not criterion2:
        criterion2 = nn.CrossEntropyLoss()

    # use SGD by default
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    # initialize list to store loss and error values
    loss_store = []
    loss_store_1, loss_store_2 = [], []
    error_store, error_store_test = [], []
    current_time = 0
    time_store = [current_time]

    for e in range(epochs):
        # initialize loss
        sum_loss = 0
        sum_loss_1, sum_loss_2 = 0, 0
        start = time.time()

        for b in range(0, train_target.size(0), mini_batch_size):
            output_classes, output_final = model(train_input.narrow(0, b * 2, mini_batch_size * 2))
            loss1 = criterion1(output_classes, train_classes.narrow(0, b * 2, mini_batch_size * 2))
            loss2 = criterion2(output_final, train_target.narrow(0, b, mini_batch_size))
            loss = w1 * loss1 + w2 * loss2
            optimizer.zero_grad()

            if w1 == 0:
                loss.backward()  # do not use intermediate loss
            else:
                loss.backward(retain_graph=True)  # retain graph allows backward pass wrt intermediate loss

            # update losses
            sum_loss = sum_loss + loss.item()
            sum_loss_1 = sum_loss_1 + loss1.item()
            sum_loss_2 = sum_loss_2 + loss2.item()

            # update gradient
            optimizer.step()

        end = time.time()
        current_time += end-start

        error_store.append(compute_nb_errors(model, train_input, train_target))

        # compute error on test set
        if not isinstance(test_input, int):
            error_store_test.append(compute_nb_errors(model, test_input, test_target))
        else:
            error_store_test = error_store
        time_store.append(current_time)
        loss_store.append(sum_loss)
        loss_store_1.append(sum_loss_1)
        loss_store_2.append(sum_loss_2)

        if verbose == 0:
            print('Epoch: {}, loss: {:0.5f}'.format(e, sum_loss))
        elif verbose == 1 and e % 5 == 0:
            print('Epoch: {}, loss: {:0.5f}'.format(e, sum_loss))
        elif verbose == 2:
            if len(error_store_test) > 0:
                print("\rTraining: {:0.1f} , Error: {}%".format(100.0*(e+1)/epochs, error_store_test[-1]), end="")
            else:
                print("\rTraining: {:0.1f} , loss: {}%".format(100.0 * (e + 1) / epochs, sum_loss), end="")

    return loss_store, time_store, loss_store_1, loss_store_2, error_store, error_store_test


def compute_nb_errors(model, input, target, mini_batch_size=100):
    """
    Funtion to compute number of errors
    Args:
        model:  model trained
        input:  tensor of size (
        target:
        mini_batch_size:

    """
    errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))

        # when using models with multiple outputs, keep the last one (target)
        if isinstance(output, tuple):
            output = output[-1]
            _, predicted_classes = output.max(1)
            errors += mini_batch_size//2 - sum(predicted_classes == target[b//2:b//2 + mini_batch_size//2])
        else:
            _, predicted_classes = output.max(1)
            errors += mini_batch_size - sum(predicted_classes == target[b:b+ mini_batch_size])
    return errors.item()


def test_model_separate(nb_h, train_i_org, train_c_org, train_t_org, test_i_org=None, test_c_org=None, test_t_org=None,
                        runs=10, epochs=25, epochs2=25, version=0, lr=1, verbose=2):

    # shuffle train
    train_i_org, train_c_org, train_t_org = shuffle(
        train_i_org.view(train_i_org.size(0) // 2, 2, train_i_org.size(2), -1),
        train_c_org.view(train_c_org.size(0) // 2, -1), train_t_org)
    idx = list(range(train_i_org.size(0)))
    max_l = train_i_org.size(0)

    nb_error_test = []
    data = {'loss_m1': [], 'loss_m2': [], 'time_m1': [], 'time_m2': [], 'nb_error_test': [], 'error_m2':[], 'error_test_m1':[], 'error_test_m2':[]}
    if version == 1:
        data['loss_m12'], data['time_m12'], data['error_test_m12'] = [], [], []

    model1 = Net_Conv(nb_h)
    model2 = Net_Full()

    for k in range(runs):
        #  if validation mode for hyperparameter tunning
        if test_i_org is None:
            test_idx = idx[max_l // 10 * k: max_l // 10 * (k + 1)]
            train_idx = [i for i in idx if i not in set(test_idx)]
            train_i, test_i = train_i_org[train_idx, :, :, :], train_i_org[test_idx, :, :, :]
            train_c, test_c = train_c_org[train_idx, :], train_c_org[test_idx, :]
            train_t, test_t = train_t_org[train_idx], train_t_org[test_idx]
            #train_t, test_t = train_t_org[train_idx, :], train_t_org[test_idx, :]
        else:
            test_i, train_i = test_i_org, train_i_org
            test_c, train_c = test_c_org, train_c_org
            test_t, train_t = test_t_org, train_t_org

        if version == 0:
            """ separate model; train with CrossEntropy Loss, train second network with raw output of first"""
            # print('Training convolutional model')
            model1 = Net_Conv(nb_h, sig=True)
            _, train_c = train_c.view(-1, 10).max(1)
            _, test_c = test_c.view(-1, 10).max(1)
            #_, train_t = train_t.view(-1,2)
            l1, t1, _, et1 = train_model(model1, train_i.view(-1, 1, train_i.size(2), train_i.size(3)), train_c.view(train_i.size(0)*2),
                                       test_input = test_i.view(-1, 1, test_i.size(2), test_i.size(3)),
                                       test_target= test_c.view(test_i.size(0) * 2),
                                       lr=lr, verbose=verbose, epochs=epochs, criterion=nn.CrossEntropyLoss())
            data['loss_m1'].append(l1)
            data['time_m1'].append(t1)
            data['error_test_m1'].append(et1)

            out_train = model1(train_i.view(-1, 1, train_i.size(2), train_i.size(3))).detach()
            out_test = model1(test_i.view(-1, 1, test_i.size(2), test_i.size(3))).detach()
            # print('Training fully connected model')
            l2, t2, e2, et2 = train_model(model2, out_train.view(-1, 20), train_t, test_input=out_test.view(-1, 20), test_target=test_t, lr=5e-1, verbose=2, epochs=epochs2)
            data['loss_m2'].append(l2)
            data['time_m2'].append(t2)
            data['error_m2'].append(e2)
            data['error_test_m2'].append(et2)

            out1 = model1(test_i.view(-1, 1, test_i.size(2), test_i.size(3)))
            #out1 = out1.detach()
            out = model2(out1.view(-1, 20).type(torch.FloatTensor))
            _, argm = out.max(1)
            #_, argm_t = test_t.max(1)
            nb_test_errors = test_t.size(0) - ((argm == test_t).sum(0))
            data['nb_error_test'].append(100.0 * nb_test_errors / test_t.size(0))
            print(' - test error {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / test_t.size(0),
                                                         nb_test_errors, test_t.size(0)))

        elif version == 1:
            """ 2 convnets + separate models; train with CrossEntropy Loss, train second network with raw output of first"""
            model11 = Net_Conv(nb_h, sig=True)
            model12 = Net_Conv(nb_h, sig=True)


            train_i_top, train_i_bottom = train_i[:, 0, :, :].view(-1, 1, 14,14), train_i[:, 1, :, :].view(-1,1,14,14)
            _, train_c_top= train_c[:, 0:10].view(-1,10).max(1)
            _, train_c_bottom = train_c[:, 10:].view(-1,10).max(1)

            test_i_top, test_i_bottom = test_i[:, 0, :, :].view(-1, 1, test_i.size(2), test_i.size(3)), \
                                        test_i[:, 1, :, :].view(-1, 1, test_i.size(2), test_i.size(3))

            _, test_c_top = test_c[:, 0:10].view(-1, 10).max(1)
            _, test_c_bottom = test_c[:, 10:].view(-1, 10).max(1)



            l11, t11, e, et11 = train_model(model11, train_i_top.view(-1, 1, train_i_top.size(2), train_i_top.size(3)), train_c_top.view(train_i_top.size(0)),
                                            test_input=test_i_top, test_target=test_c_top,
                                         lr=lr, verbose=verbose, epochs=epochs, criterion=nn.CrossEntropyLoss())

            l12, t12, e, et12 = train_model(model12, train_i_bottom.view(-1, 1, train_i_bottom.size(2), train_i_bottom.size(3)),
                        train_c_bottom.view(train_i_bottom.size(0)), test_input=test_i_bottom, test_target=test_c_bottom, lr=lr,
                        verbose=verbose, epochs=epochs, criterion=nn.CrossEntropyLoss())
            data['loss_m1'].append(l11)
            data['time_m1'].append(t11)
            data['error_test_m1'].append(et11)
            data['loss_m12'].append(l12)
            data['time_m12'].append(t12)
            data['error_test_m12'].append(et12)

            out_train_top = model11(train_i_top.view(-1, 1, train_i_top.size(2), train_i_top.size(3))).detach()
            out_train_bottom = model12(train_i_bottom.view(-1, 1, train_i_bottom.size(2), train_i_bottom.size(3))).detach()

            out_test_top = model11(test_i_top.view(-1, 1, test_i_top.size(2), test_i_top.size(3))).detach()
            out_test_bottom = model12(
                test_i_bottom.view(-1, 1, test_i_bottom.size(2), test_i_bottom.size(3))).detach()

            out_train = torch.stack((out_train_top, out_train_bottom), dim=1).view(-1,20)
            out_test = torch.stack((out_test_top, out_test_bottom), dim=1).view(-1, 20)
            # print('Training fully connected model')
            l2, t2, e2, et2 = train_model(model2, out_train.view(-1, 20), train_t, out_test.view(-1, 20), test_t,
                                          lr=5e-1, verbose=verbose, epochs=epochs2)
            data['loss_m2'].append(l2)
            data['time_m2'].append(t2)
            data['error_m2'].append(e2)
            data['error_test_m2'].append(et2)



            out11 = model11(test_i_top.view(-1, 1, test_i.size(2), test_i.size(3)))
            out12 = model12(test_i_bottom.view(-1, 1, test_i.size(2), test_i.size(3)))
            out_test = torch.stack((out11, out12), dim=1).view(-1, 20)
            #out1 = out1.detach()
            out = model2(out_test.view(-1, 20).type(torch.FloatTensor))
            _, argm = out.max(1)
            #_, argm_t = test_t.max(1)
            nb_test_errors = test_t.size(0) - ((argm == test_t).sum(0))
            data['nb_error_test'].append(100.0 * nb_test_errors / test_t.size(0))
            print(' - test error {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / test_t.size(0),
                                                         nb_test_errors, test_t.size(0)))


    p1 = sum([params.numel() for params in model1.parameters()])
    p2 = sum([params.numel() for params in model2.parameters()])
    print('Model with {} parameters \t'.format(p1 + p2))
    errors = torch.tensor(data['nb_error_test']).type(torch.FloatTensor)
    print('Mean error: {:0.2f} Std deviation in error: {:0.2f}'.format(errors.mean(), errors.std()))
    return data



def test_model_joint(train_i_org, train_c_org, train_t_org, test_i_org=None, test_c_org=None, test_t_org=None, runs=10,
                     epochs=25, w1=1, w2=1, verbose=2, lr=0.5):
    # shuffle train
    train_i_org, train_c_org, train_t_org = shuffle(
        train_i_org.view(train_i_org.size(0) // 2, 2, train_i_org.size(2), -1),
        train_c_org.view(train_c_org.size(0) // 2, -1), train_t_org)
    idx = list(range(train_i_org.size(0)))
    max_l = train_i_org.size(0)

    data = {'loss': [], 'nb_error_test':[], 'time':[], 'loss_m1':[], 'loss_m2':[], 'error_m2':[], 'error_test_m2':[]}

    for k in range(runs):
        #model = Net_All(16, 32, 64, 100)
        model = Net_small_all(2**6, 2**6, 2**4)

        #  if validation mode for hyperparameter tunning
        if test_i_org is None:
            test_idx = idx[max_l // 10 * k: max_l // 10 * (k + 1)]
            train_idx = [i for i in idx if i not in set(test_idx)]
            train_i, test_i = train_i_org[train_idx, :, :, :], train_i_org[test_idx, :, :, :]
            train_c, test_x = train_c_org[train_idx, :], train_c_org[test_idx, :]
            #train_t, test_t = train_t_org[train_idx, :], train_t_org[test_idx, :]
            train_t, test_t = train_t_org[train_idx], train_t_org[test_idx]
        else:
            test_i, train_i = test_i_org, train_i_org
            test_c, train_c = test_c_org, train_c_org
            test_t, train_t = test_t_org, train_t_org

        _, train_c = train_c.view(-1, 10).max(1)
        #_, train_t = train_t.max(1)
        #  train model
        l, t, l1, l2, e2, et2 = train_model_all(model, train_i.view(-1, 1, train_i.size(2), train_i.size(3)), train_c.view(train_i.size(0)*2), train_t,
                            lr=lr, verbose=verbose, epochs=epochs, w1=w1, w2=w2,
                            test_input=test_i.view(-1, 1, test_i.size(2), test_i.size(3)), test_target=test_t)

        data['loss'].append(l)
        data['time'].append(t)
        data['loss_m1'].append(l1)
        data['loss_m2'].append(l2)
        data['error_m2'].append(e2)
        data['error_test_m2'].append(et2)

        # get output from test
        out_class, out_target = model(test_i.view(-1, 1, test_i.size(2), test_i.size(3)))
        _, argmax_class = out_class.max(1)
        _, pred = out_target.max(1)
        #_, argm_t = test_t.max(1)
        nb_test_errors = test_t.size(0) - (pred == test_t).sum(0)
        data['nb_error_test'].append(100.0 * nb_test_errors / test_t.size(0))
        print(' - test error Net {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / test_t.size(0),
                                                         nb_test_errors, test_t.size(0)))

    p = sum([params.numel() for params in model.parameters()])
    print('Model with {} parameters'.format(p))
    errors = torch.tensor(data['nb_error_test']).type(torch.FloatTensor)
    print('Mean error: {:0.2f} Std deviation in error: {:0.2f}'.format(errors.mean(), errors.std()))
    return data




def test_model_fc(train_i_org, train_c_org, train_t_org, test_i_org=None, test_c_org=None,test_t_org=None, runs=10,
                     epochs=100, l_rate=0.5, verbose=2):
    # shuffle train
    train_i_org, train_c_org, train_t_org = shuffle(
        train_i_org.view(train_i_org.size(0) // 2, 2, train_i_org.size(2), -1),
        train_c_org.view(train_c_org.size(0) // 2, -1), train_t_org)
    idx = list(range(train_i_org.size(0)))
    max_l = train_i_org.size(0)

    data = {'loss': [], 'time': [], 'nb_error_test': [], 'error':[], 'error_test':[]}

    for k in range(runs):
        #  if validation mode for hyperparameter tunning
        if test_i_org is None:
            test_idx = idx[max_l // 10 * k: max_l // 10 * (k + 1)]
            train_idx = [i for i in idx if i not in set(test_idx)]
            train_i, test_i = train_i_org[train_idx, :, :, :], train_i_org[test_idx, :, :, :]
            #train_t, test_t = train_t_org[train_idx, :], train_t_org[test_idx, :]
            train_t, test_t = train_t_org[train_idx], train_t_org[test_idx]
        else:
            test_i, train_i = test_i_org, train_i_org
            test_t, train_t = test_t_org, train_t_org

        # define model
        model = Net_fc(train_i.size(2) * train_i.size(3) * 2)

        #_, train_t = train_t.max(1)
        #  train model
        l, t, e, et = train_model(model, train_i.view(-1, 2*train_i.size(2)*train_i.size(3)), train_t,
                                  test_i.view(-1, 2*test_i.size(2)*test_i.size(3)), test_t, lr=l_rate, verbose=verbose, epochs=epochs)

        data['loss'].append(l)
        data['time'].append(t)
        data['error'].append(e)
        data['error_test'].append(et)

        # get output from test
        out_target = model(test_i.view(-1, 2*test_i.size(2)*test_i.size(3)))
        _, pred = out_target.max(1)
        #_, argm_t = test_t.max(1)
        nb_test_errors = test_t.size(0) - (pred == test_t).sum(0)
        data['nb_error_test'].append(100.0 * nb_test_errors / test_t.size(0))
        print(' - test error Net {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / test_t.size(0),
                                                         nb_test_errors, test_t.size(0)))

    p = sum([params.numel() for params in model.parameters()])
    print('Model with {} parameters'.format(p))
    errors = torch.tensor(data['nb_error_test']).type(torch.FloatTensor)
    print('Mean error: {:0.2f} Std deviation in error: {:0.2f}'.format(errors.mean(), errors.std()))
    return data

