import matplotlib.pyplot as plt
import torch

def generate_graphic_loss(loss, data, save=False):
    l = torch.tensor(loss)
    loss_m = l.mean(0)
    loss_std = l.std(0)
    xdata = list(range(l.size(1)))

    plt.figure()
    plt.title(data['title'])
    plt.xlabel('Epochs');
    plt.ylabel('Loss')
    plt.plot(xdata, loss_m.tolist(), 'r')
    plt.fill_between(xdata, torch.clamp(loss_m - loss_std, min=0).tolist(), (loss_m + loss_std).tolist(), color='gray', alpha=0.2)
    if save:
        plt.savefig('output/{}.pdf'.format(data['file_name']), bbox_inches='tight')
    plt.show()
