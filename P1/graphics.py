import matplotlib.pyplot as plt
import torch

def generate_graphic_loss(loss, data, time=None, save=False):
    """
    Function to generate a plot of the loss for a single model
    Args:
        loss: list containing as many lists as runs. Each inner list contains the loss along the train.
            size = [[n_epochs],[n_epochs],...,[n_epochs]](n_runs)
        data: dictionary containing title, file_name, x_axis name
        time: list of lists (same sizes as loss) containing the end time of each epoch.
        save: boolean. If true, save the plot

    """
    l = torch.tensor(loss)
    loss_m = l.mean(0)
    loss_std = l.std(0)
    if not time:
        xdata = list(range(l.size(1)))
    else:
        xdata = torch.tensor(time).mean(0)[1:].tolist()

    plt.figure()
    plt.title(data['title'])
    plt.xlabel(data['x_axis'])
    plt.ylabel('Loss')
    plt.plot(xdata, loss_m.tolist(), 'r')
    plt.fill_between(xdata, torch.clamp(loss_m - loss_std, min=0).tolist(), (loss_m + loss_std).tolist(), color='gray', alpha=0.2)
    if save:
        plt.savefig('output/{}.pdf'.format(data['file_name']), bbox_inches='tight')
    plt.show()


def generate_multiple_graphic_loss(loss_list, data, time=None, save=False):
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
        fig, ax = plt.subplots(1,2)

        plt.subplots_adjust(wspace=0.9)
        # ax[0] --> plot loss vs. epochs
        # ax[1] --> plot loss vs. time

        ax[0].set_title(data['title'])
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel(data['y_axis'])
        ax[1].set_title(data['title'])
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel(data['y_axis'])
        for i, loss in enumerate(loss_list):

            l = torch.tensor(loss).type(torch.FloatTensor)
            loss_m = l.mean(0) # compute mean along different runs for a given model
            loss_std = l.std(0) # compute std along different runs for a given model
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

        ax[0].legend(data['legend'])
        ax[1].legend(data['legend'])

    else: # if no time vector is provided plot only loss vs. epochs
        fig, ax = plt.subplots(1,1)
        ax.set_title(data['title'])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(data['y_axis'])
        for i, loss in enumerate(loss_list):
            l = torch.tensor(loss).type(torch.FloatTensor)
            loss_m = l.mean(0)  # compute mean along different runs for a given model
            loss_std = l.std(0) # compute std along different runs for a given model
            xdata_epochs = list(range(l.size(1)))
            # plot mean as line
            ax.plot(xdata_epochs, loss_m.tolist())
            # fill region [mean + std, mean - std]
            ax.fill_between(xdata_epochs, torch.clamp(loss_m - loss_std, min=0).tolist(),
                               (loss_m + loss_std).tolist(),
                               alpha=0.2)

        ax.legend(data['legend'])


    if save:
        plt.savefig('output/{}.pdf'.format(data['file_name']), bbox_inches='tight')
    plt.show()