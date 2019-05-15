from Optimizers import DecreaseSGD
from data_generation import plot_result, plot_loss


def compute_number_error(output_one_hot, target_one_hot):
    """

    :param output_one_hot: tensor of size [nb, 2] generated by the model (class in one-hot coding)
    :param target_one_hot: tensor of size [nb, 2] of true ground label (class in one-hot coding)
    :return: (int) number of errors
    """
    output = output_one_hot.argmax(dim=1)
    target = target_one_hot.argmax(dim=1)
    nb_of_error = (output != target).sum()
    return nb_of_error

def train_model(model,train_input, train_target, optimizer, epochs_number, Sample_number, batch_size):
    """
    Train model
    :param model: Sequence instance
    :param train_input: tensor of size [Sample_number, 2] with coordinates x,y of the samples
    :param train_target: tensor of size [nb] with labels of the samples
    :param optimizer: instance of Optimizers class
    :param epochs_number: (int) number of iterations along the whole train set
    :param Sample_number: (int) Size of training set
    :param batch_size: (int) Size of batch to perform SGD
    :return: my_loss: (list) values of loss along the epochs
    """
    # Train model
    my_loss = []
    for epochs in range(0, epochs_number):
        for b in range(0, Sample_number, batch_size):
            output = model.forward(train_input.narrow(0, b, batch_size))
            loss_value = model.compute_loss(output, train_target.narrow(0, b, batch_size))
            model.backward()

            if isinstance(optimizer, DecreaseSGD):
                optimizer(epochs, model.sequence)
            else:
                optimizer(model.sequence)

        if epochs % 50 == 0:
            print("\rEpochs {}. Loss: {:0.5f}".format(epochs, loss_value.item()), end="")
        my_loss.append(loss_value.item())
    return my_loss



def evaluate_model(model, train_input, train_target, test_input, test_target, loss, mname=None):
    """
    Evaluate performance of the model in terms of errors and generate plots
    :param model: instance of Sequence. Model already trained!
    :param train_input: tensor of size [Sample_number, 2] with coordinates x,y of the samples
    :param train_target: tensor of size [nb] with labels of the samples
    :param test_input: tensor of size [Sample_number_test, 2] with coordinates x,y of the samples
    :param test_target: tensor of size [nb] with labels of the samples
    :param loss: list of losses along epochs during the training
    :param mname: File name. If None, the figure is not saved. Otherwise the figure is saved in the output folder

    """
    # Evalute Model in train set
    epochs_number = len(loss)
    output = model.forward(train_input)
    l = model.compute_loss(output, train_target)

    print("\nLoss: ", l.item())
    print("Number of errors: ", compute_number_error(output, train_target).item())

    id_class_train = output.argmax(dim=1)
    plot_result(train_input, train_target, id_class_train, fname=mname)
    plot_loss(range(0, epochs_number), loss, fname=mname)

    # Evaluate Model in test set
    output = model.forward(test_input)
    l = model.compute_loss(output, test_target)

    print("\nLoss: ", l.item())
    print("Number of errors: ", compute_number_error(output, test_target).item())


    id_class_test = output.argmax(dim=1)
    plot_result(test_input, test_target, id_class_test, train=False, fname=mname)