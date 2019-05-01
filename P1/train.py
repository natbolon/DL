import torch
from torch import nn, optim

from generate_data import shuffle, to_one_hot
from models import Net_All, Net_Conv, Net_Full, Net_small_all, Net_fc


# Define training function for two-stage model

def train_model(model, train_input, train_target, epochs=25, \
                mini_batch_size=100, lr=1e-3, criterion=None, optimizer=None, verbose=2):
    # use MSE loss by default
    if not criterion:
        criterion = nn.MSELoss()

    # use SGD by default
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    loss_store = []

    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.item()
            optimizer.step()
        loss_store.append(sum_loss)
        if verbose == 0:
            print('Epoch: {}, loss: {:0.2f}'.format(e, sum_loss))
        elif verbose == 1 and e % 5 == 0:
            print(e, sum_loss)

    return loss_store


#  Define training function for complete model
def train_model_all(model, train_input, train_classes, train_target, epochs=25, \
                    mini_batch_size=100, lr=1e-3, w1=1, w2=1, criterion1=None, criterion2=None, optimizer=None,
                    verbose=2):
    print('Training Composed model')

    # use MSE loss by default
    if not criterion1:
        criterion1 = nn.MSELoss()
    if not criterion2:
        criterion2 = nn.MSELoss()

    # use SGD by default
    if not optimizer:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_store = []
    for e in range(epochs):
        sum_loss = 0

        for b in range(0, train_target.size(0), mini_batch_size):
            output_classes, output_final = model(train_input.narrow(0, b * 2, mini_batch_size * 2))
            loss1 = criterion1(output_classes, train_classes.narrow(0, b * 2, mini_batch_size * 2))
            loss2 = criterion2(output_final, train_target.narrow(0, b, mini_batch_size))
            loss = w1 * loss1 + w2 * loss2
            model.zero_grad()
            loss.backward(retain_graph=True)
            sum_loss = sum_loss + loss.item()
            optimizer.step()
        loss_store.append(sum_loss)

        if verbose == 0:
            print('Epoch: {}, loss: {:0.2f}'.format(e, sum_loss))
        elif verbose == 1 and e % 5 == 0:
            print(e, sum_loss)

    return loss_store


def compute_nb_errors(model, input, target, mini_batch_size=100):
    errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)

        for k in range(mini_batch_size):
            if target.data[b + k, predicted_classes[k]] <= 0:
                errors = errors + 1
    return errors


def test_model_separate(nb_h, train_i_org, train_c_org, train_t_org, test_i_org=None, test_c_org=None, test_t_org=None,
                        runs=10, epochs=25, version=0):

    # shuffle train
    train_i_org, train_c_org, train_t_org = shuffle(
        train_i_org.view(train_i_org.size(0) // 2, 2, train_i_org.size(2), -1),
        train_c_org.view(train_c_org.size(0) // 2, -1), train_t_org)
    idx = list(range(train_i_org.size(0)))
    max_l = train_i_org.size(0)

    nb_error_test = []
    loss_store = []

    model1 = Net_Conv(nb_h)
    model2 = Net_Full()

    for k in range(runs):
        #  if validation mode for hyperparameter tunning
        if test_i_org is None:
            test_idx = idx[max_l // 10 * k: max_l // 10 * (k + 1)]
            train_idx = [i for i in idx if i not in set(test_idx)]
            train_i, test_i = train_i_org[train_idx, :, :, :], train_i_org[test_idx, :, :, :]
            train_c, test_x = train_c_org[train_idx, :], train_c_org[test_idx, :]
            train_t, test_t = train_t_org[train_idx, :], train_t_org[test_idx, :]
        else:
            test_i, train_i = test_i_org, train_i_org
            test_c, train_c = test_c_org, train_c_org
            test_t, train_t = test_t_org, train_t_org

        if version == 0:
            """ separate model; train with MSE loss, one hot encoding for output to feed as input"""
            # print('Training convolutional model')
            train_model(model1, train_i.view(-1, 1, train_i.size(2), train_i.size(3)), train_c.view(-1, 10), lr=1,
                        verbose=2, epochs=epochs)
            # print('Training fully connected model')
            loss_store.append(train_model(model2, train_c.view(-1, 20), train_t, lr=5e-1, verbose=2))

            _, out1 = model1(test_i.view(-1, 1, test_i.size(2), test_i.size(3))).max(1)
            out = model2(to_one_hot(out1.view(-1, 1)).view(-1, 20))
            _, argm = out.max(1)
            _, argm_t = test_t.max(1)
            nb_test_errors = test_t.size(0) - ((argm == argm_t).sum(0))
            nb_error_test.append(100.0 * nb_test_errors / test_t.size(0))
            print('test error {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / test_t.size(0),
                                                         nb_test_errors, test_t.size(0)))

        elif version == 1:
            """ separate model; train with MSE loss, train second network with raw output of first"""
            # print('Training convolutional model')
            train_model(model1, train_i.view(-1, 1, train_i.size(2), train_i.size(3)), train_c.view(-1, 10), lr=1,
                        verbose=2, epochs=epochs)

            out_train = model1(train_i.view(-1, 1, train_i.size(2), train_i.size(3))).detach()
            # print('Training fully connected model')
            loss_store.append(train_model(model2, out_train.view(-1, 20), train_t, lr=5e-1, verbose=2))

            _, out1 = model1(test_i.view(-1, 1, test_i.size(2), test_i.size(3))).max(1)
            out = model2(to_one_hot(out1.view(-1, 1)).view(-1, 20))
            _, argm = out.max(1)
            _, argm_t = test_t.max(1)
            nb_test_errors = test_t.size(0) - ((argm == argm_t).sum(0))
            nb_error_test.append(100.0 * nb_test_errors / test_t.size(0))
            print('test error {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / test_t.size(0),
                                                         nb_test_errors, test_t.size(0)))

        elif version == 2:
            """ separate model; train with CrossEntropy Loss, train second network with raw output of first"""
            # print('Training convolutional model')
            _, train_c = train_c.view(-1, 10).max(1)
            train_model(model1, train_i.view(-1, 1, train_i.size(2), train_i.size(3)), train_c.view(train_i.size(0)*2), lr=1e-3,
                        verbose=2, epochs=epochs, criterion=nn.CrossEntropyLoss())

            out_train = model1(train_i.view(-1, 1, train_i.size(2), train_i.size(3))).detach()
            # print('Training fully connected model')
            loss_store.append(train_model(model2, out_train.view(-1, 20), train_t, lr=5e-1, verbose=2))

            out1 = model1(test_i.view(-1, 1, test_i.size(2), test_i.size(3)))
            #out1 = out1.detach()
            out = model2(out1.view(-1, 20).type(torch.FloatTensor))
            _, argm = out.max(1)
            _, argm_t = test_t.max(1)
            nb_test_errors = test_t.size(0) - ((argm == argm_t).sum(0))
            nb_error_test.append(100.0 * nb_test_errors / test_t.size(0))
            print('test error {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / test_t.size(0),
                                                         nb_test_errors, test_t.size(0)))


    p1 = sum([params.numel() for params in model1.parameters()])
    p2 = sum([params.numel() for params in model2.parameters()])
    print('Model with {} parameters \t'.format(p1 + p2))
    errors = torch.tensor(nb_error_test).type(torch.FloatTensor)
    print('Mean error: {:0.2f} Std deviation in error: {:0.2f}'.format(errors.mean(), errors.std()))
    return nb_error_test, loss_store



def test_model_joint(train_i_org, train_c_org, train_t_org, test_i_org=None, test_c_org=None, test_t_org=None, runs=10,
                     epochs=25):
    # shuffle train
    train_i_org, train_c_org, train_t_org = shuffle(
        train_i_org.view(train_i_org.size(0) // 2, 2, train_i_org.size(2), -1),
        train_c_org.view(train_c_org.size(0) // 2, -1), train_t_org)
    idx = list(range(train_i_org.size(0)))
    max_l = train_i_org.size(0)

    nb_error_test = []
    loss = []

    for k in range(runs):
        #model = Net_All(16, 32, 64, 100)
        model = Net_small_all(2**4, 2**7, 2**5)

        #  if validation mode for hyperparameter tunning
        if test_i_org is None:
            test_idx = idx[max_l // 10 * k: max_l // 10 * (k + 1)]
            train_idx = [i for i in idx if i not in set(test_idx)]
            train_i, test_i = train_i_org[train_idx, :, :, :], train_i_org[test_idx, :, :, :]
            train_c, test_x = train_c_org[train_idx, :], train_c_org[test_idx, :]
            train_t, test_t = train_t_org[train_idx, :], train_t_org[test_idx, :]
        else:
            test_i, train_i = test_i_org, train_i_org
            test_c, train_c = test_c_org, train_c_org
            test_t, train_t = test_t_org, train_t_org

        #  train model
        loss.append(
            train_model_all(model, train_i.view(-1, 1, train_i.size(2), train_i.size(3)), train_c.view(-1, 10), train_t,
                            lr=0.5, verbose=2, epochs=epochs))

        # get output from test
        out_class, out_target = model(test_i.view(-1, 1, test_i.size(2), test_i.size(3)))
        _, argmax_class = out_class.max(1)
        _, pred = out_target.max(1)
        _, argm_t = test_t.max(1)
        nb_test_errors = argm_t.size(0) - (pred == argm_t).sum(0)
        nb_error_test.append(100.0 * nb_test_errors / argm_t.size(0))
        print('test error Net {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / argm_t.size(0),
                                                         nb_test_errors, argm_t.size(0)))

    p = sum([params.numel() for params in model.parameters()])
    print('Model with {} parameters'.format(p))
    errors = torch.tensor(nb_error_test).type(torch.FloatTensor)
    print('Mean error: {:0.2f} Std deviation in error: {:0.2f}'.format(errors.mean(), errors.std()))
    return nb_error_test, loss




def test_model_fc(train_i_org, train_c_org, train_t_org, test_i_org=None, test_c_org=None,test_t_org=None, runs=10,
                     epochs=100, l_rate=0.5):
    # shuffle train
    train_i_org, train_c_org, train_t_org = shuffle(
        train_i_org.view(train_i_org.size(0) // 2, 2, train_i_org.size(2), -1),
        train_c_org.view(train_c_org.size(0) // 2, -1), train_t_org)
    idx = list(range(train_i_org.size(0)))
    max_l = train_i_org.size(0)

    nb_error_test = []
    loss = []

    for k in range(runs):
        #  if validation mode for hyperparameter tunning
        if test_i_org is None:
            test_idx = idx[max_l // 10 * k: max_l // 10 * (k + 1)]
            train_idx = [i for i in idx if i not in set(test_idx)]
            train_i, test_i = train_i_org[train_idx, :, :, :], train_i_org[test_idx, :, :, :]
            train_t, test_t = train_t_org[train_idx, :], train_t_org[test_idx, :]
        else:
            test_i, train_i = test_i_org, train_i_org
            test_t, train_t = test_t_org, train_t_org

        # define model
        model = Net_fc(train_i.size(2) * train_i.size(3) * 2)

        #  train model
        loss.append(
            train_model(model, train_i.view(-1, 2*train_i.size(2)*train_i.size(3)), train_t,
                            lr=l_rate, verbose=1, epochs=epochs))

        # get output from test
        out_target = model(test_i.view(-1, 2*test_i.size(2)*test_i.size(3)))
        _, pred = out_target.max(1)
        _, argm_t = test_t.max(1)
        nb_test_errors = argm_t.size(0) - (pred == argm_t).sum(0)
        nb_error_test.append(100.0 * nb_test_errors / argm_t.size(0))
        print('test error Net {:0.2f}% {:d}/{:d}'.format((100.0 * nb_test_errors) / argm_t.size(0),
                                                         nb_test_errors, argm_t.size(0)))

    p = sum([params.numel() for params in model.parameters()])
    print('Model with {} parameters'.format(p))
    errors = torch.tensor(nb_error_test).type(torch.FloatTensor)
    print('Mean error: {:0.2f} Std deviation in error: {:0.2f}'.format(errors.mean(), errors.std()))
    return nb_error_test, loss

