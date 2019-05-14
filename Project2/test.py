import matplotlib.pyplot as plt

import torch
torch.set_grad_enabled(False)

from Linear import Linear
from Activation import Tanh, Relu, Sigmoid
from Loss import LossMSE, CrossEntropy
from Optimizers import Optimizers, Sgd, DecreaseSGD, Adam
from Sequential import Sequential
from data_generation import generate_disc_set, plot_disc
from evalute_models import compute_number_error, evaluate_model, train_model

def main():
    # Generate data
    print('Generate data')
    Sample_number = 1000
    train_input, train_target = generate_disc_set(Sample_number)
    test_input, test_target = generate_disc_set(Sample_number)

    # Visualize data
    plot_disc(train_input, train_target, "Train data before normalization")

    # Normalize data
    mu, std = train_input.mean(0), train_input.std(0)
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    # Plot train and test samples after normalization
    plot_disc(train_input, train_target, "Train data after normalization")
    plot_disc(test_input, test_target, "Test data after normalization")

    # Define constants along the test
    hidden_nb = 25
    std = 0.1
    eta = 3e-1
    batch_size = 200
    epochs_number = 1000


    
    # Model 1. No dropout; constant learning rate (SGD)
    print('\nModel 1: Optimizer: SGD; No dropout; ReLU; CrossEntropy')

    # Define model name for plots
    mname='Model1'

    # Define structure of the network
    linear_1 = Linear(2, hidden_nb)
    relu_1 = Relu()
    linear_2 = Linear(hidden_nb, hidden_nb)
    relu_2 = Relu()
    linear_3 = Linear(hidden_nb, hidden_nb)
    relu_3 = Relu()
    linear_4 = Linear(hidden_nb, 2)
    loss = CrossEntropy()

    model_1 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=CrossEntropy())

    # Initialize weights
    model_1.normalize_parameters(mean=0, std=std)
    # Define optimizer
    optimizer = Sgd(eta)

    # Train model
    my_loss_1 = train_model(model_1, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)

    # Evalute model and produce plots
    evaluate_model(model_1, train_input, train_target, test_input, test_target, my_loss_1, mname=mname)


    # Model 2. No dropout; decreasing learning rate (DecreaseSGD)
    print('\nModel 2: Optimizer: DecreaseSGD; No dropout; ReLU; CrossEntropy')

    # Define model name for plots
    mname='Model2'

    # Define structure of the network
    linear_1 = Linear(2, hidden_nb)
    relu_1 = Relu()
    linear_2 = Linear(hidden_nb, hidden_nb)
    relu_2 = Relu()
    linear_3 = Linear(hidden_nb, hidden_nb)
    relu_3 = Relu()
    linear_4 = Linear(hidden_nb, 2)

    model_2 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=CrossEntropy())

    # Initialize weights
    model_2.normalize_parameters(mean=0, std=std)
    # Define optimizer
    optimizer = DecreaseSGD(eta)

    # Train model
    my_loss_2 = train_model(model_2, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)
    # Evalute model and produce plots
    evaluate_model(model_2, train_input, train_target, test_input, test_target, my_loss_2, mname=mname)

    # Model 3. No dropout; Adam Optimizer
    print('\nModel 3: Optimizer: Adam; No dropout; ReLU; CrossEntropy')

    # Define model name for plots
    mname='Model3'

    # Custom hyperparameters
    eta_adam=1e-3
    epochs_number_adam = 500

    # Define structure of the network
    linear_1 = Linear(2, hidden_nb)
    relu_1 = Relu()
    linear_2 = Linear(hidden_nb, hidden_nb)
    relu_2 = Relu()
    linear_3 = Linear(hidden_nb, hidden_nb)
    relu_3 = Relu()
    linear_4 = Linear(hidden_nb, 2)
    loss = CrossEntropy()

    model_3 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=CrossEntropy())

    # Initialize weights
    model_3.normalize_parameters(mean=0, std=std)
    # Define optimizer
    optimizer = Adam(eta_adam, 0.9, 0.99, 1e-8)

    # Train model
    my_loss_3 = train_model(model_3, train_input, train_target, optimizer, epochs_number_adam, Sample_number, batch_size)

    # Evalute model and produce plots
    evaluate_model(model_3, train_input, train_target, test_input, test_target, my_loss_3,  mname=mname)


    # PLOT TO COMPARE OPTIMIZERS
    plt.figure(figsize=(10,4))
    plt.plot(range(0, epochs_number), my_loss_1, linewidth=1)
    plt.plot(range(0, epochs_number), my_loss_2, linewidth=1)
    plt.plot(range(0, epochs_number_adam), my_loss_3, linewidth=1)
    plt.legend(["SGD", "Decreasing SGD", "Adam"])
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig('output/compare_optimizers.pdf', bbox_inches='tight')
    plt.show()


    # Model 4. Dropout; SGD
    print('\nModel 4: Optimizer: SGD; Dropout; ReLU; CrossEntropy')

    # Define model name for plots
    mname='Model4'

    # Define structure of the network
    dropout = 0.25

    linear_1 = Linear(2, hidden_nb)
    relu_1 = Relu()
    linear_2 = Linear(hidden_nb, hidden_nb, dropout=dropout)
    relu_2 = Relu()
    linear_3 = Linear(hidden_nb, hidden_nb, dropout=dropout)
    relu_3 = Relu()
    linear_4 = Linear(hidden_nb, 2)

    model_4 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=CrossEntropy())

    # Initialize weights
    model_4.normalize_parameters(mean=0, std=std)
    # Define optimizer
    optimizer = Sgd(eta)

    # Train model
    my_loss_4 = train_model(model_4, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)

    # Evalute model and produce plots
    evaluate_model(model_4, train_input, train_target, test_input, test_target, my_loss_4,  mname=mname)


    # PLOT TO COMPARE DROPOUT AND NO DROPOUT
    plt.figure(figsize=(10,4))
    plt.plot(range(0, epochs_number), my_loss_1, linewidth=1)
    plt.plot(range(0, epochs_number), my_loss_4, linewidth=1)
    plt.legend(["Without Dropout", "With Dropout"])
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig('output/compare_dropout.pdf', bbox_inches='tight')
    plt.show()


    print('\nEvaluation of different activation functions\n')

    # Model 5. No Dropout; SGD; Tanh
    print('\nModel 5: Optimizer: SGD; No dropout; Tanh; CrossEntropy')

    # Define model name for plots
    mname='Model5'

    # Define structure of the network
    linear_1 = Linear(2, hidden_nb)
    relu_1 = Tanh()
    linear_2 = Linear(hidden_nb, hidden_nb)
    relu_2 = Tanh()
    linear_3 = Linear(hidden_nb, hidden_nb)
    relu_3 = Tanh()
    linear_4 = Linear(hidden_nb, 2)

    model_5 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=CrossEntropy())

    # Initialize weights
    model_5.normalize_parameters(mean=0, std=std)
    # Define optimizer
    optimizer = Sgd(eta)

    # Train model
    my_loss_5 = train_model(model_5, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)

    # Evalute model and produce plots
    evaluate_model(model_5, train_input, train_target, test_input, test_target, my_loss_5,  mname=mname)


    # Model 6. Xavier Initialization
    print('\nModel 6: Optimizer: SGD; No dropout; Tanh; Xavier initialization; CrossEntropy')

    # Define model name for plots
    mname='Model6'

    # Define network structure
    linear_1 = Linear(2, hidden_nb)
    relu_1 = Tanh()
    linear_2 = Linear(hidden_nb, hidden_nb)
    relu_2 = Tanh()
    linear_3 = Linear(hidden_nb, hidden_nb)
    relu_3 = Tanh()
    linear_4 = Linear(hidden_nb, 2)

    model_6 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=CrossEntropy())

    model_6.xavier_parameters()
    optimizer = Sgd()

    # Train model
    my_loss_6 = train_model(model_6, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)

    # Evalute model and produce plots
    evaluate_model(model_6, train_input, train_target, test_input, test_target, my_loss_6,  mname=mname)


    # Model 7. Sigmoid
    print('\nModel 7: Optimizer: SGD; No dropout; Sigmoid; CrossEntropy')

    # Define model name for plots
    mname='Model7'

    # Define parameter for sigmoid activation
    p_lambda = 0.1

    # Define network structure
    linear_1 = Linear(2, hidden_nb)
    relu_1 = Sigmoid(p_lambda)
    linear_2 = Linear(hidden_nb, hidden_nb)
    relu_2 = Sigmoid(p_lambda)
    linear_3 = Linear(hidden_nb, hidden_nb)
    relu_3 = Sigmoid(p_lambda)
    linear_4 = Linear(hidden_nb, 2)

    model_7 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=CrossEntropy())

    model_7.normalize_parameters(mean=0.5, std=1)
    optimizer = Sgd(eta=0.5)

    # Train model
    my_loss_7 = train_model(model_7, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)

    # Evalute model and produce plots
    evaluate_model(model_7, train_input, train_target, test_input, test_target, my_loss_7,  mname=mname)


    
    # PLOT TO COMPARE EFFECT OF DIFFERENT ACTIVATIONS
    plt.figure(figsize=(10,4))
    plt.plot(range(0, epochs_number), my_loss_1, linewidth=0.5)
    plt.plot(range(0, epochs_number), my_loss_5, linewidth=0.5, alpha=0.8)
    plt.plot(range(0, epochs_number), my_loss_6, linewidth=0.5, alpha=0.8)
    plt.plot(range(0, epochs_number), my_loss_7,  linewidth=0.5)
    plt.legend(["Relu", "Tanh", "Tanh (Xavier)", "Sigmoid"])
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig('output/compare_activations.pdf', bbox_inches='tight')
    plt.show()

    print('\nEvaluation of base model with MSE loss\n')

    # Model 8. MSE loss
    print('\nModel 8: Optimizer: SGD; No dropout; Relu; MSE')

    # Define model name for plots
    mname='Model8'
    linear_1 = Linear(2, hidden_nb)
    relu_1 = Relu()
    linear_2 = Linear(hidden_nb, hidden_nb)
    relu_2 = Relu()
    linear_3 = Linear(hidden_nb, hidden_nb)
    relu_3 = Relu()
    linear_4 = Linear(hidden_nb, 2)
    loss = LossMSE()

    model_8 = Sequential(linear_1, relu_1, linear_2, relu_2, linear_3, relu_3, linear_4, loss=loss)

    model_8.normalize_parameters(mean=0, std=std)
    optimizer = Sgd(eta)

    # Train model
    my_loss_8 = train_model(model_8, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)

    # Evalute model and produce plots
    evaluate_model(model_8, train_input, train_target, test_input, test_target, my_loss_8,  mname=mname)


    print('Evaluation done! ')

if __name__=='__main__':
    main()
