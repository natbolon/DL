from random import seed
import matplotlib.pyplot as plt
import torch

from torch.autograd import Variable

from dlc_practical_prologue import generate_pair_sets
from generate_data import shuffle, normalize_data
from graphics import generate_multiple_graphic_loss, generate_graphic_two_models
from train import test_model_fc, test_model_separate, test_model_joint

def run():

    # Hide plots
    plt.ioff()
    print('ATTENTION: PLOTS WILL NOT BE SHOWN. \nALL OF THEM ARE STOREM IN THE OUTPUT FOLDER')

    # set seed for reproducibility purposes
    seed(28)

    print('Generate data')
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

    # shuffle train set
    train_input, train_classes, train_target = shuffle(train_input, train_classes, train_target)

    # reshape
    train_input = train_input.reshape((train_input.size(0) * 2, 1, train_input.size(2), train_input.size(3)))
    test_input = test_input.reshape((test_input.size(0) * 2, 1, test_input.size(2), test_input.size(3)))

    train_classes = train_classes.view(-1,1)
    test_classes = test_classes.view(-1,1)

    # normalize
    normalize_data(train_input, test_input)


    # Generate variables
    print('Generate Variables')
    train_input, train_target, train_classes = Variable(train_input), Variable(train_target), Variable(train_classes)
    test_input, test_target, test_classes = Variable(test_input), Variable(test_target), Variable(test_classes)

    # Define common parameters

    r = 10 # number of runs
    v = 3  # verbose


    print('\n --- Evaluate models for hyperparameter tunning ---')
    # Hyperparameter tunning for model 1.1
    print('\nHyperparameter tunning for Model 1.1')
    e = 500

    print('training Model 1.1 with lr=0.5\n')
    data11_4 = test_model_fc(train_input, train_classes, train_target, runs=r, epochs=e, l_rate=0.5 * 1e-1, verbose=v)
    print('training Model 1.1 with lr=1e-1\n')
    data11_1 = test_model_fc(train_input, train_classes, train_target, runs=r, epochs=e, l_rate=1e-1, verbose=v)
    print('training Model 1.1 with lr=1e-2\n')
    data11_2 = test_model_fc(train_input, train_classes, train_target, runs=r, epochs=e, l_rate=1e-2, verbose=v)
    print('training Model 1.1 with lr=1e-3\n')
    data11_3 = test_model_fc(train_input, train_classes, train_target, runs=r, epochs=e, l_rate=1e-3, verbose=v)


    # Generate graphics
    legend = ['lr=0.5', 'lr=1e-1','lr=1e-2','lr=1e-3']
    g_names = {'file_name': 'train_m11_hyperparameters_epochs500', 'title': 'Validation Error for target prediction', 'y_axis': 'Errors' , 'legend': legend}

    m = [data11_4, data11_1, data11_2, data11_3]
    generate_multiple_graphic_loss([mod['error_test'] for mod in m], g_names, save=True)


    # Hyperparameter tunning for model 2.1
    print('\nHyperparameter tunning for Model 2.1')
    e21 = 50

    print('training Model 2.1 with lr(1)=1 and lr(2)=0.5\n')
    data21_1 = test_model_separate(train_input, train_classes, train_target, runs=r, epochs=e21, epochs2=15, version=0, lr=1, verbose=v)
    print('training Model 2.1 with lr(1)=0.5 and lr(2)=0.5\n')
    data21_2 = test_model_separate(train_input, train_classes, train_target, runs=r, epochs=e21, epochs2=15, version=0, lr=0.5, verbose=v)
    print('training Model 2.1 with lr(1)=0.1 and lr(2)=0.5\n')
    data21_3 = test_model_separate(train_input, train_classes, train_target, runs=r, epochs=e21, epochs2=15, version=0, lr=0.1, verbose=v)
    print('training Model 2.1 with lr(1)=1e-2 and lr(2)=0.5\n')
    data21_4 = test_model_separate(train_input, train_classes, train_target, runs=r, epochs=e21, epochs2=15, version=0, lr=1e-2, verbose=v)
    print('training Model 2.1 with lr(1)=0.5 and lr(2)=0.5\n')
    data21_5 = test_model_separate(train_input, train_classes, train_target, runs=10, epochs=20, epochs2=25, version=0, lr=0.5, verbose=v)
    print('training Model 2.1 with lr(1)=0.5 and lr(2)=0.1\n')
    data21_6 = test_model_separate(train_input, train_classes, train_target, runs=10, epochs=20, epochs2=25, version=0, lr=0.5,lr2=0.1, verbose=v)

    # Generate graphic
    m = [data21_1, data21_2, data21_3 ,data21_4]
    mm = [data21_5, data21_6]
    legend = ['lr=1','lr=0.5','lr=1e-1', 'lr=1e-2']
    legend2 = ['lr=0.5','lr=1e-1']

    g_names = {'file_name': 'train_m21_hyperparameters_error_m1_rows', 'title': ['Validation Error for class prediction', 'Validation Error for target prediction'],
               'y_axis': 'Errors' , 'legend': legend, 'legend_2': legend2}
    generate_graphic_two_models([mod['error_test_m1'] for mod in m],[mod['error_test_m2'] for mod in mm], g_names, version='rows', save=True)

    # Weight tunning for model 3.1
    print('\nWeight tunning for Model 3.1')
    e31=120
    l_rate = 0.5
    print('\ntraining Model 3.1 with w1=0.7 and w2=0.3\n')
    data31_07 = test_model_joint(train_input, train_classes, train_target, runs=r, epochs=e31, lr=l_rate, w1=0.7, w2=0.3, verbose=v)
    print('\ntraining Model 3.1 with w1=0.3 and w2=0.7\n')
    data31_03 = test_model_joint(train_input, train_classes, train_target, runs=r, epochs=e31, lr=l_rate, w1=0.3, w2=0.7, verbose=v)
    print('\ntraining Model 3.1 with w1=0.5 and w2=0.5\n')
    data31_05 = test_model_joint(train_input, train_classes, train_target, runs=r, epochs=e31, lr=l_rate, w1=0.5, w2=0.5, verbose=v)


    # Generate graphic
    m = [data31_07, data31_03, data31_05]
    legend = ['w1=0.7, w2=0.3', 'w1=0.3, w2=0.7', 'w1=w2=0.5']
    g_names = {'file_name': 'train_m31_weights', 'title': 'Validation Error for target prediction', 'y_axis': 'Errors' , 'legend': legend}
    generate_multiple_graphic_loss([mod['error_test_m2'] for mod in m], g_names, save=True)

    ### EVALUATE THE DIFFERENT MODELS IN THE TEST SET
    print('\n----Evaluate models with TEST set----')

    print('\nEvaluate model 1.1')
    # evaluate model 1.1 in TEST set
    d_model_11 = test_model_fc(train_input, train_classes, train_target,test_i_org=test_input, test_c_org=test_classes, test_t_org=test_target, runs=r, epochs=220, l_rate=1e-2, verbose=3)

    print('\nEvaluate model 1.2')
    # evaluate model 1.2 in TEST set
    d_model_12 = test_model_joint(train_input, train_classes, train_target, test_input, test_classes, test_target, runs=r, epochs=35, w1=0, lr=0.1, verbose=3)

    print('\nEvaluate model 2.1')
    # evaluate model 2.1 in TEST set
    d_model_21 = test_model_separate(train_input, train_classes, train_target, test_input, test_classes, test_target, runs=r, epochs=30, epochs2=25, version=0, lr=0.1, verbose=2)

    print('\nEvaluate model 2.2')
    # evaluate model 2.2 in TEST set
    d_model_22 = test_model_separate(train_input, train_classes, train_target, test_input, test_classes, test_target, runs=r, epochs=30, epochs2=25, version=1, lr=0.1, verbose=2)

    print('\nEvaluate model 3.1')
    # evaluate model 3.1 in TEST set
    d_model_31 = test_model_joint(train_input, train_classes, train_target, test_input, test_classes, test_target, runs=r, epochs=55, lr=0.5, verbose=2, w1=0.7, w2=0.3)

    # generate graphic to compare single models in terms of errors along epochs and time
    m = [d_model_11, d_model_12, d_model_31]
    l = ['Model 1.1','Model 1.2', 'Model 3.1']

    g_names = {'file_name': 'comparison_single_models', 'title': 'Test Error for target prediction', 'y_axis': 'Errors' , 'legend': l}
    generate_multiple_graphic_loss([ d_model_11['error_test'], d_model_12['error_test_m2'], d_model_31['error_test_m2']], g_names, [mod['time'] for mod in m],version='rows',save=True)

    print('DONE!')

if __name__ == "__main__":
    run()