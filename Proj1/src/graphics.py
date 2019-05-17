import matplotlib.pyplot as plt
import torch


def generate_graphic_two_models(loss1, loss2, data, version='columns', save=False):
    """
    Function to generate a plot of the errors/loss for models 2.1 or 2.2
    :param loss1: list containing as many lists as runs. Each inner list contains the loss along the train.
            size = [[n_epochs],[n_epochs],...,[n_epochs]](n_runs)
    :param loss2: list containing as many lists as runs. Each inner list contains the loss along the train.
            size = [[n_epochs],[n_epochs],...,[n_epochs]](n_runs)
    :param data: dictionary containing title, file_name, x_axis name
    :param version: 'columns' if subplots aligned in the same column, otherwise, aligned in the same row.
    :param save: boolean. If true, save the plot
    :return: plot of loss along epochs and/or time
    """

    if version == 'columns':
        fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    else:
        fig, ax = plt.subplots(2, 1, figsize=(15, 9))
        plt.subplots_adjust(hspace=0.5)

    ax[0].set_title(data['title'][0], fontsize=22)
    ax[0].set_xlabel('Epochs', fontsize=22)
    ax[0].set_ylabel(data['y_axis'], fontsize=22)

    ax[1].set_title(data['title'][1], fontsize=22)
    ax[1].set_xlabel('Epochs', fontsize=22)
    ax[1].set_ylabel(data['y_axis'], fontsize=22)

    for i, loss in enumerate(loss1):
        l = torch.tensor(loss).type(torch.FloatTensor)
        loss_m = l.mean(0)  # compute mean along different runs for a given model
        loss_std = l.std(0)  # compute std along different runs for a given model
        xdata_epochs = list(range(l.size(1)))
        # plot mean as line
        ax[0].plot(xdata_epochs, loss_m.tolist())
        # fill region [mean + std, mean - std]
        ax[0].fill_between(xdata_epochs, torch.clamp(loss_m - loss_std, min=0).tolist(),
                           (loss_m + loss_std).tolist(),
                           alpha=0.2)

    for i, loss in enumerate(loss2):
        l = torch.tensor(loss).type(torch.FloatTensor)
        loss_m = l.mean(0)  # compute mean along different runs for a given model
        loss_std = l.std(0)  # compute std along different runs for a given model
        xdata_epochs = list(range(l.size(1)))
        # plot mean as line
        ax[1].plot(xdata_epochs, loss_m.tolist())
        # fill region [mean + std, mean - std]
        ax[1].fill_between(xdata_epochs, torch.clamp(loss_m - loss_std, min=0).tolist(),
                           (loss_m + loss_std).tolist(),
                           alpha=0.2)

    ax[0].tick_params(labelsize=16)
    ax[0].legend(data['legend'], fontsize=14)

    ax[1].tick_params(labelsize=16)
    ax[1].legend(data['legend_2'], fontsize=14)

    if save:
        plt.savefig('../output/{}.pdf'.format(data['file_name']), bbox_inches='tight')
    plt.close(fig)




def generate_multiple_graphic_loss(loss_list, data, time=None, version='columns', save=False):
    """
    Function to generate a plot of the loss for a single model
    :param loss_list: list containing as many lists as models. Each model lists contains as many lists as runs.
                        Each inner list contains the loss along the train.
                    size = [[[n_epochs],[n_epochs],...,[n_epochs]](n_runs), ..., [[n_epochs],[n_epochs],...,[n_epochs]](n_runs)] (n_models)
    :param data: dictionary containing title, file_name, y_axis name
    :param time: list of lists of lists (same sizes as loss) containing the end time of each epoch.
                if None, a single plot is generated.
                else, two subplots are generated: loss vs. epochs and loss vs. time
    :param version:  'columns' if subplots aligned in the same column, otherwise, aligned in the same row.
    :param save: boolean. If true, save the plot
    :return: plot of losses along epochs and/or time
    """

    if time is not None:
        if version == 'columns':
            fig, ax = plt.subplots(1, 2, figsize=(18, 4))  # figsize = (w, h)
        else:
            fig, ax = plt.subplots(2, 1, figsize=(18, 12))  # figsize = (w, h)

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        # ax[0] --> plot loss vs. epochs
        # ax[1] --> plot loss vs. time
        fs = 22

        ax[0].set_title(data['title'], fontsize=fs)
        ax[0].set_xlabel('Epochs', fontsize=fs)
        ax[0].set_ylabel(data['y_axis'], fontsize=fs)
        ax[1].set_title(data['title'], fontsize=fs)
        ax[1].set_xlabel('Time [s]', fontsize=fs)
        ax[1].set_ylabel(data['y_axis'], fontsize=fs)
        for i, loss in enumerate(loss_list):
            l = torch.tensor(loss).type(torch.FloatTensor)
            loss_m = l.mean(0)  # compute mean along different runs for a given model
            loss_std = l.std(0)  # compute std along different runs for a given model
            xdata_epochs = list(range(l.size(1)))
            xdata = torch.tensor(time[i]).mean(0)[1:].tolist()
            # plot mean as line
            ax[0].plot(xdata_epochs, loss_m.tolist())
            ax[1].plot(xdata, loss_m.tolist())
            # fill region [mean + std, mean - std]
            ax[0].fill_between(xdata_epochs, torch.clamp(loss_m - loss_std, min=0).tolist(),
                               (loss_m + loss_std).tolist(),
                               alpha=0.2)
            ax[1].fill_between(xdata, torch.clamp(loss_m - loss_std, min=0).tolist(), (loss_m + loss_std).tolist(),
                               alpha=0.2)

        ax[0].tick_params(labelsize=16)
        ax[1].tick_params(labelsize=16)
        ax[0].legend(data['legend'], fontsize=14)
        ax[1].legend(data['legend'], fontsize=14)




    else:  # if no time vector is provided plot only loss vs. epochs

        fs = 22
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.set_title(data['title'], fontsize=fs)
        ax.set_xlabel('Epochs', fontsize=fs)
        ax.set_ylabel(data['y_axis'], fontsize=fs)
        for i, loss in enumerate(loss_list):
            l = torch.tensor(loss).type(torch.FloatTensor)
            loss_m = l.mean(0)  # compute mean along different runs for a given model
            loss_std = l.std(0)  # compute std along different runs for a given model
            xdata_epochs = list(range(l.size(1)))
            # plot mean as line
            ax.plot(xdata_epochs, loss_m.tolist())
            # fill region [mean + std, mean - std]
            ax.fill_between(xdata_epochs, torch.clamp(loss_m - loss_std, min=0).tolist(),
                            (loss_m + loss_std).tolist(),
                            alpha=0.2)

        ax.tick_params(labelsize=16)
        ax.legend(data['legend'], fontsize=14)

    if save:
        plt.savefig('../output/{}.pdf'.format(data['file_name']), bbox_inches='tight')
    plt.close(fig)


def error_comparison(m11, m12, m21, m22, m31, save=True):
    """
    Generate boxplot of error on test set and time required for training  for each of the models
    :param m11: dictionary of data from model 1.1
    :param m12: dictionary of data from model 1.2
    :param m21: dictionary of data from model 2.1
    :param m22: dictionary of data from model 2.2
    :param m31: dictionary of data from model 3.1
    :param save: boolean. save the plot in the folder output
    """

    # Define time for each model
    t_m11 = [m11['time'][i][-1] for i in range(len(m11['time']))]
    t_m12 = [m12['time'][i][-1] for i in range(len(m12['time']))]

    t_m21 = [m21['time_m1'][i][-1] + m21['time_m2'][i][-1] for i in range(len(m21['time_m1']))]
    t_m22 = [m22['time_m1'][i][-1] + m22['time_m2'][i][-1] + m22['time_m12'][i][-1] for i in range(len(m22['time_m1']))]

    t_m31 = [m31['time'][i][-1] for i in range(len(m31['time']))]

    # list of times
    times = [t_m11, t_m12, t_m21, t_m22, t_m31]

    # labels for plot
    labs = ['Model 1.1', 'Model 1.2', 'Model 2.1', 'Model 2.2', 'Model 3.1']

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].boxplot([d['nb_error_test'] for d in [m11, m12, m21, m22, m31]], labels=labs)
    ax[0].set_title('Error on test set', fontsize=20)
    ax[0].set_ylabel('Error percentage', fontsize=16)

    ax[1].boxplot(times, labels=labs)
    ax[1].set_title('Training time', fontsize=20)
    ax[1].set_ylabel('Time [s]', fontsize=16)
    if save:
        plt.savefig('../output/{}.pdf'.format('error_boxplot'), bbox_inches='tight')
    plt.close(fig)