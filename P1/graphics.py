import matplotlib.pyplot as plt
import torch

def generate_graphic_loss(loss, data, time=None, save=False):
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
    if time is not None:
        fig, ax = plt.subplots(1,2)

        plt.subplots_adjust(wspace=0.9)
        #plot vs epochs

        ax[0].set_title(data['title'])
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel(data['y_axis'])
        ax[1].set_title(data['title'])
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel(data['y_axis'])
        for i, loss in enumerate(loss_list):
            l = torch.tensor(loss).type(torch.FloatTensor)
            loss_m = l.mean(0)
            loss_std = l.std(0)
            xdata_epochs = list(range(l.size(1)))
            xdata = torch.tensor(time[i]).mean(0)[1:].tolist()
            ax[0].plot(xdata_epochs, loss_m.tolist())
            ax[1].plot(xdata, loss_m.tolist())
            ax[0].fill_between(xdata_epochs, torch.clamp(loss_m - loss_std, min=0).tolist(),
                               (loss_m + loss_std).tolist(),
                               alpha=0.2)
            ax[1].fill_between(xdata, torch.clamp(loss_m - loss_std, min=0).tolist(), (loss_m + loss_std).tolist(),
                               alpha=0.2)

        ax[0].legend(data['legend'])
        ax[1].legend(data['legend'])
    else:
        fig, ax = plt.subplots(1,1)
        ax.set_title(data['title'])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(data['y_axis'])
        for i, loss in enumerate(loss_list):
            l = torch.tensor(loss).type(torch.FloatTensor)
            loss_m = l.mean(0)
            loss_std = l.std(0)
            xdata_epochs = list(range(l.size(1)))
            ax.plot(xdata_epochs, loss_m.tolist())
            ax.fill_between(xdata_epochs, torch.clamp(loss_m - loss_std, min=0).tolist(),
                               (loss_m + loss_std).tolist(),
                               alpha=0.2)

        ax.legend(data['legend'])


    if save:
        plt.savefig('output/{}.pdf'.format(data['file_name']), bbox_inches='tight')
    plt.show()