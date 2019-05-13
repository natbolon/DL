import matplotlib.pyplot as plt
import torch


def generate_graphic_two_models(loss1, loss2, data, version='columns', save=False):
    """
    Function to generate a plot of the errors/loss for models 2.1 or 2.2
    Args:
        loss: list containing as many lists as runs. Each inner list contains the loss along the train.
            size = [[n_epochs],[n_epochs],...,[n_epochs]](n_runs)
        data: dictionary containing title, file_name, x_axis name
        time: list of lists (same sizes as loss) containing the end time of each epoch.
        save: boolean. If true, save the plot

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
    plt.show()




def generate_multiple_graphic_loss(loss_list, data, time=None, version='columns', save=False):
    """
        Function to generate a plot of the loss for a single model
        Args:
            loss_list: list containing as many lists as models. Each model lists contains as many lists as runs.
                        Each inner list contains the loss along the train.
                    size = [[[n_epochs],[n_epochs],...,[n_epochs]](n_runs), ..., [[n_epochs],[n_epochs],...,[n_epochs]](n_runs)] (n_models)
            data: dictionary containing title, file_name, y_axis name
            time: list of lists of lists (same sizes as loss) containing the end time of each epoch.
                if None, a single plot is generated.
                else, two subplots are generated: loss vs. epochs and loss vs. time
            save: boolean. If true, save the plot

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
    plt.show()