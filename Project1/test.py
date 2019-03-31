import torch
import dlc_practical_prologue as prologue
import torch.nn as nn
import matplotlib.pyplot as plt #To remove (use matplotlib)


def main():
    N = 1000 #num
    train_input, train_target, train_classes, test_input, test_target, test_classes = import_data(N, True)
    
    return 0

def import_data(N, normalize):
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    if (normalize):
        train_input = (train_input - train_input.mean())/train_input.std()
        test_input = (test_input - train_input.mean())/train_input.std()
	
    return (train_input, train_target, train_classes, test_input, test_target, test_classes)


def print_image(im):        #To remove (use matplotlib)
    plt.imshow(im, cmap='gray')
    plt.show()
    
    




if __name__ == '__main__':
    main()
