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


def run_all_model(train_input, train_target, test_input, test_target, Sample_number, save_plot=False):

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
    model_1_perf = evaluate_model(model_1, train_input, train_target, test_input, test_target, my_loss_1, save_plot, mname=mname)

    
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
    model_2_perf = evaluate_model(model_2, train_input, train_target, test_input, test_target, my_loss_2, save_plot, mname=mname)

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
    model_3_perf = evaluate_model(model_3, train_input, train_target, test_input, test_target, my_loss_3, save_plot,  mname=mname)

	
    # PLOT TO COMPARE OPTIMIZERS
    if save_plot:
        fig = plt.figure(figsize=(10,4))
        plt.plot(range(0, epochs_number), my_loss_1, linewidth=1)
        plt.plot(range(0, epochs_number), my_loss_2, linewidth=1)
        plt.plot(range(0, epochs_number_adam), my_loss_3, linewidth=1)
        plt.legend(["SGD", "Decreasing SGD", "Adam"])
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.savefig('output/compare_optimizers.pdf', bbox_inches='tight')
        plt.close(fig)


    # Model 4. Dropout; SGD
    print('\nModel 4: Optimizer: Decreasing SGD; Dropout; ReLU; CrossEntropy')

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
    optimizer = DecreaseSGD(eta)

    # Train model
    my_loss_4 = train_model(model_4, train_input, train_target, optimizer, epochs_number, Sample_number, batch_size)

    # Evalute model and produce plots
    model_4_perf = evaluate_model(model_4, train_input, train_target, test_input, test_target, my_loss_4, save_plot,  mname=mname)


    # PLOT TO COMPARE DROPOUT AND NO DROPOUT
    if save_plot:
        fig = plt.figure(figsize=(10,4))
        plt.plot(range(0, epochs_number), my_loss_2, linewidth=1)
        plt.plot(range(0, epochs_number), my_loss_4, linewidth=1)
        plt.legend(["Without Dropout", "With Dropout"])
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.savefig('output/compare_dropout.pdf', bbox_inches='tight')
        plt.close(fig)


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
    model_5_perf = evaluate_model(model_5, train_input, train_target, test_input, test_target, my_loss_5, save_plot,  mname=mname)


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
    model_6_perf = evaluate_model(model_6, train_input, train_target, test_input, test_target, my_loss_6, save_plot, mname=mname)


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
    model_7_perf = evaluate_model(model_7, train_input, train_target, test_input, test_target, my_loss_7, save_plot,  mname=mname)


    
    # PLOT TO COMPARE EFFECT OF DIFFERENT ACTIVATIONS
    if  save_plot:
        fig = plt.figure(figsize=(10,4))
        plt.plot(range(0, epochs_number), my_loss_1, linewidth=0.5)
        plt.plot(range(0, epochs_number), my_loss_5, linewidth=0.5, alpha=0.8)
        plt.plot(range(0, epochs_number), my_loss_6, linewidth=0.5, alpha=0.8)
        plt.plot(range(0, epochs_number), my_loss_7,  linewidth=0.5)
        plt.legend(["Relu", "Tanh", "Tanh (Xavier)", "Sigmoid"])
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.savefig('output/compare_activations.pdf', bbox_inches='tight')
        plt.close(fig)

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
    model_8_perf = evaluate_model(model_8, train_input, train_target, test_input, test_target, my_loss_8, save_plot,  mname=mname)


    print('Evaluation done! ')
    
    train_loss  = torch.tensor([model_1_perf[0], model_2_perf[0], model_3_perf[0], model_4_perf[0], model_5_perf[0], model_6_perf[0], model_7_perf[0], model_8_perf[0]])
    train_error = torch.tensor([model_1_perf[1], model_2_perf[1], model_3_perf[1], model_4_perf[1], model_5_perf[1], model_6_perf[1], model_7_perf[1], model_8_perf[1]])
    test_loss   = torch.tensor([model_1_perf[2], model_2_perf[2], model_3_perf[2], model_4_perf[2], model_5_perf[2], model_6_perf[2], model_7_perf[2], model_8_perf[2]])
    test_error  = torch.tensor([model_1_perf[3], model_2_perf[3], model_3_perf[3], model_4_perf[3], model_5_perf[3], model_6_perf[3], model_7_perf[3], model_8_perf[3]])
    
    return train_loss, train_error, test_loss, test_error

def main():
    nb_of_run = 2
    Sample_number = 1000
    
    train_loss = torch.empty((8, nb_of_run))
    train_error = torch.empty((8, nb_of_run))
    test_loss = torch.empty((8, nb_of_run))
    test_error = torch.empty((8, nb_of_run))
    
    # Do NOT show plots
    plt.ioff()

    print('ATTENTION: PLOTS WILL NOT BE SHOWN. \nALL OF THEM ARE STORED IN THE OUTPUT FOLDER')

    # Generate data
    print('Generate data')
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
    
    
    
    
    
    #First run: plot are saved
    print("Run number 1 /", nb_of_run, ":\n")
    out = run_all_model(train_input, train_target, test_input, test_target, Sample_number, save_plot=True)
    train_loss[:,0] = out[0]
    train_error[:,0] = out[1]
    test_loss[:,0] = out[2]
    test_error[:,0] = out[3]
    
    train_loss = train_loss.type(torch.float)
    test_loss = test_loss.type(torch.float)
    train_error = train_error.type(torch.float)
    test_error = test_error.type(torch.float)
    
    
    
    #For other run: no need to save plot
    for i in range(1, nb_of_run):
        print("\n\n\nRun number ",i+1 ,"/", nb_of_run, ":\n")
        out = run_all_model(train_input, train_target, test_input, test_target, Sample_number, save_plot=False)
        train_loss[:,i] = out[0]
        train_error[:,i] = out[1]
        test_loss[:,i] = out[2]
        test_error[:,i] = out[3]
	
    #Boxplot of losses and error
    fig = plt.figure(figsize=(10,5))
    plt.boxplot(train_loss, labels=["model 1", "model 2", "model 3", "model 4", "model 5", "model 6", "model 7", "model 8"])
    plt.title('Loss on train set', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.savefig('output/{}.pdf'.format('train_loss_boxplot'), bbox_inches='tight')
    
    fig = plt.figure(figsize=(10,5))
    plt.boxplot(train_error.mul_(100).div_(Sample_number), labels=["model 1", "model 2", "model 3", "model 4", "model 5", "model 6", "model 7", "model 8"])
    plt.title('Error on train set', fontsize=20)
    plt.ylabel('Error percentage', fontsize=20)
    plt.savefig('output/{}.pdf'.format('train_error_boxplot'), bbox_inches='tight')
    
    fig = plt.figure(figsize=(10,5))
    plt.boxplot(test_loss, labels=["model 1", "model 2", "model 3", "model 4", "model 5", "model 6", "model 7", "model 8"])
    plt.title('Loss on test set', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.savefig('output/{}.pdf'.format('test_loss_boxplot'), bbox_inches='tight')
    
    fig = plt.figure(figsize=(10,5))
    plt.boxplot(test_error.mul_(100).div_(Sample_number), labels=["model 1", "model 2", "model 3", "model 4", "model 5", "model 6", "model 7", "model 8"])
    plt.title('Error on test set', fontsize=20)
    plt.ylabel('Error percentage', fontsize=20)
    plt.savefig('output/{}.pdf'.format('test_error_boxplot'), bbox_inches='tight')
    
    print("\nmodel 1:")
    print("train loss mean: ", train_loss[0,:].mean().item())
    print("train error mean: ", train_error[0,:].mean().item())
    print("test loss mean: ", test_loss[0,:].mean().item())
    print("test error mean: ", test_error[0,:].mean().item())
    
    print("\nmodel 2:")
    print("train loss mean: ", train_loss[1,:].mean().item())
    print("train error mean: ", train_error[1,:].mean().item())
    print("test loss mean: ", test_loss[1,:].mean().item())
    print("test error mean: ", test_error[1,:].mean().item())
    
    print("\nmodel 3:")
    print("train loss mean: ", train_loss[2,:].mean().item())
    print("train error mean: ", train_error[2,:].mean().item())
    print("test loss mean: ", test_loss[2,:].mean().item())
    print("test error mean: ", test_error[2,:].mean().item())
    
    print("\nmodel 4:")
    print("train loss mean: ", train_loss[3,:].mean().item())
    print("train error mean: ", train_error[3,:].mean().item())
    print("test loss mean: ", test_loss[3,:].mean().item())
    print("test error mean: ", test_error[3,:].mean().item())
    
    print("\nmodel 5:")
    print("train loss mean: ", train_loss[4,:].mean().item())
    print("train error mean: ", train_error[4,:].mean().item())
    print("test loss mean: ", test_loss[4,:].mean().item())
    print("test error mean: ", test_error[4,:].mean().item())
    
    print("\nmodel 6:")
    print("train loss mean: ", train_loss[5,:].mean().item())
    print("train error mean: ", train_error[5,:].mean().item())
    print("test loss mean: ", test_loss[5,:].mean().item())
    print("test error mean: ", test_error[5,:].mean().item())
    
    print("\nmodel 7:")
    print("train loss mean: ", train_loss[6,:].mean().item())
    print("train error mean: ", train_error[6,:].mean().item())
    print("test loss mean: ", test_loss[6,:].mean().item())
    print("test error mean: ", test_error[6,:].mean().item())
    
    print("\nmodel 8:")
    print("train loss mean: ", train_loss[7,:].mean().item())
    print("train error mean: ", train_error[7,:].mean().item())
    print("test loss mean: ", test_loss[7,:].mean().item())
    print("test error mean: ", test_error[7,:].mean().item())
    
if __name__=='__main__':
    main()
