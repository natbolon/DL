from dlc_practical_prologue import generate_pair_sets
from generate_data import binarize, shuffle, normalize_data, to_one_hot
from torch.autograd import Variable

from train import test_model_separate, test_model_joint, test_model_fc
from graphics import generate_graphic_loss, generate_multiple_graphic_loss

print('Generate data')
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

# convert to binary target
train_target_bin, test_target_bin = binarize(train_target), binarize(test_target)

# shuffle train set
#train_input, train_classes, train_target_bin = shuffle(train_input, train_classes, train_target_bin)
train_input, train_classes, train_target = shuffle(train_input, train_classes, train_target)

# reshape
train_input = train_input.reshape((train_input.size(0) * 2, 1, train_input.size(2), train_input.size(3)))
test_input = test_input.reshape((test_input.size(0) * 2, 1, test_input.size(2), test_input.size(3)))

# normalize
normalize_data(train_input, test_input)

# convert to one hot encoding
train_classes_one_hot = to_one_hot(train_classes.view(train_classes.size(0) * 2, -1))
test_classes_one_hot = to_one_hot(test_classes.view(test_classes.size(0) * 2, -1))

print('Generate Variables')
train_input, train_target, train_classes_one_hot, train_classes = Variable(train_input), Variable(train_target), Variable(train_classes_one_hot), Variable(train_classes)
test_input, test_target, test_classes_one_hot, test_classes = Variable(test_input), Variable(test_target), Variable(test_classes_one_hot), Variable(test_classes)

"""
#EVALUATE MLP WITH SINGLE LOSS (model 1.1)

# print('FC model - Train evaluation')
# _, loss = test_model_fc(train_input, train_classes_one_hot, train_target_bin, runs=10, epochs=60, l_rate=0.1)
#g_names = {'file_name': 'train_fc_model', 'title': 'Train Loss'}
#generate_graphic_loss(loss, g_names, save=False)

#print('FC model - Train evaluation')
#_, loss = test_model_fc(train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin,  runs=10, epochs=40)
#g_names = {'file_name': 'train_fc_model', 'title': 'Train Loss'}
#generate_graphic_loss(loss, g_names, save=False)


# EVALUATE JOINT MODEL WITHOUT INTERMEDIATE LOSS (model 1.2)

# print('Joint model - Train evaluation')
# errors, loss = test_model_joint(train_input, train_classes_one_hot, train_target_bin, epochs=50, w1=0, lr=0.1)
# g_names = {'file_name': 'train_joint', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=False)

# print('Joint model - Test evaluation')
# errors, loss = test_model_joint(train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin,  epochs=50, w1=0, lr=0.1)
# g_names = {'file_name': 'test_joint', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=True)
"""

# TRAIN MODELS FOR HYPER-PARAMETER TESTING
r = 10

# MODEL 1.1
""" for model 1.1 --> lr=0.1; epochs = 40"""
#TODO : REPEAT WITH 500 EPOCHS! 
e = 500

data11 = test_model_fc(train_input, train_classes_one_hot, train_target, runs=r, epochs=e, l_rate=1e-1, verbose=2)
data12 = test_model_fc(train_input, train_classes_one_hot, train_target, runs=r, epochs=e, l_rate=1e-2, verbose=2)
data31 = test_model_fc(train_input, train_classes_one_hot, train_target, runs=r, epochs=e, l_rate=1e-3, verbose=2)
g_names = {'file_name': 'train_m11_hyperparameters_epochs500', 'title': 'Test Error for target prediction', 'y_axis': 'Errors' , 'legend': ['lr=1e-1','lr=1e-2','lr=1e-3']}
generate_multiple_graphic_loss([data11['error_test'], data12['error_test'], data31['error_test']], g_names, save=True)


# MODEL 1.2
""" for model 1.2 --> lr=0.5; epochs = 25
e12 = 75

data12_1 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=r, epochs=e12, w1=0, lr=1, verbose=2)
data12_2 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=r, epochs=e12, w1=0, lr=5*1e-1, verbose=2)
data12_3 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=r, epochs=e12, w1=0, lr=1e-1, verbose=2)


g_names = {'file_name': 'train_m12_hyperparameters', 'title': 'Test Error for target prediction', 'y_axis': 'Errors' , 'legend': ['lr=1', 'lr=0.5', 'lr=0.1']}
generate_multiple_graphic_loss([data12_1['error_test_m2'], data12_2['error_test_m2'],data12_3['error_test_m2']], g_names, save=True)
"""


# MODEL 2.1
""" For model 2.1 --> lr=1 for convnet and 0.5 for MLP; epochs=25 for convnet and 15 for MLP
e21 = 50

data21_1 = test_model_separate(112, train_input, train_classes_one_hot, train_target, runs=r, epochs=e21, epochs2=15, version=0, lr=1, verbose=2)
data21_2 = test_model_separate(112, train_input, train_classes_one_hot, train_target, runs=r, epochs=e21, epochs2=15, version=0, lr=0.5, verbose=2)
data21_3 = test_model_separate(112, train_input, train_classes_one_hot, train_target, runs=r, epochs=e21, epochs2=15, version=0, lr=0.1, verbose=2)

g_names = {'file_name': 'train_m21_hyperparameters_class', 'title': 'Test Error for class prediction', 'y_axis': 'Errors' , 'legend': ['lr=1', 'lr=0.5', 'lr=0.1']}
generate_multiple_graphic_loss([data21_1['error_test_m1'], data21_2['error_test_m1'],data21_3['error_test_m1']], g_names, save=True)


g_names = {'file_name': 'train_m22_hyperparameters_target', 'title': 'Test Error for target prediction', 'y_axis': 'Errors' , 'legend': ['lr=1', 'lr=0.5', 'lr=0.1']}
generate_multiple_graphic_loss([data21_1['error_test_m2'], data21_2['error_test_m2'],data21_3['error_test_m2']], g_names, save=True)

"""


# MODEL 2.2
"""FOR MODEL 2.2 --> lr=0.5, epochs=30 lr=1; epochs2=15 
e22 = 75
data22_1 = test_model_separate(112, train_input, train_classes_one_hot, train_target, runs=r, epochs=e22, epochs2=15, version=1, lr=1, verbose=2)
data22_2 = test_model_separate(112, train_input, train_classes_one_hot, train_target, runs=r, epochs=e22, epochs2=15, version=1, lr=0.5, verbose=2)
data22_3 = test_model_separate(112, train_input, train_classes_one_hot, train_target, runs=r, epochs=e22, epochs2=15, version=1, lr=0.1, verbose=2)

g_names = {'file_name': 'train_m22_hyperparameters_class11_long', 'title': 'Test Error for class prediction', 'y_axis': 'Errors' , 'legend': ['lr=1', 'lr=0.5', 'lr=0.1']}
#generate_multiple_graphic_loss([data21_1['error_test_m1'], data21_2['error_test_m1'],data21_3['error_test_m1']], g_names, [data21_1['time_m1'], data21_2['time_m1'], data21_3['time_m1']], save=False)
generate_multiple_graphic_loss([data22_1['error_test_m1'], data22_2['error_test_m1'],data22_3['error_test_m1']], g_names, save=True)

g_names = {'file_name': 'train_m22_hyperparameters_class12_long', 'title': 'Test Error for class prediction', 'y_axis': 'Errors' , 'legend': ['lr=1', 'lr=0.5', 'lr=0.1']}
generate_multiple_graphic_loss([data22_1['error_test_m12'], data22_2['error_test_m12'],data22_3['error_test_m12']], g_names, save=True)

g_names = {'file_name': 'train_m22_hyperparameters_target_long', 'title': 'Test Error for target prediction', 'y_axis': 'Errors' , 'legend': ['lr=1', 'lr=0.5', 'lr=0.1']}
generate_multiple_graphic_loss([data22_1['error_test_m2'], data22_2['error_test_m2'],data22_3['error_test_m2']], g_names, save=True)

"""





# MODEL 3.1
"""
e31=75
data31_1 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=r, epochs=e31, lr=1, verbose=2)
data31_2 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=r, epochs=e31, lr=5*1e-1, verbose=2)
data31_3 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=r, epochs=e31, lr=1e-1, verbose=2)
g_names = {'file_name': 'train_m31_hyperparameters', 'title': 'Test Error for target prediction', 'y_axis': 'Errors' , 'legend': ['lr=1e-1','lr=1e-2','lr=1e-3']}
generate_multiple_graphic_loss([data31_1['error_test_m2'], data31_2['error_test_m2'], data31_3['error_test_m2']], g_names, save=True)

"""



########
########
########

# # convnet with single loss
# print('Training model 1.2')
data12 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=5, epochs=25, w1=0, lr=0.5, verbose=2)
# # convnet with intermediate loss
# print('Training model 3.1')
data31 = test_model_joint(train_input, train_classes_one_hot, train_target, runs=5, epochs=25, lr=0.1, verbose=2)
# # multiple models: 1 convnet + 1 MLP
# print('Training model 2.1')
#data21 = test_model_separate(112, train_input, train_classes_one_hot, train_target, runs=5, epochs=25, version=1, verbose=2)
# multiple models: 2 convnet + 1 MLP
print('Training model 2.2')
#data2 = test_model_separate(112, train_input, train_classes_one_hot, train_target_bin, runs=5, epochs=25, version=0)

g_names = {'file_name': 'train_separate_m2', 'title': 'Train Error for target prediction', 'y_axis': 'Errors' , 'legend': ['Model 1.2', 'Model 3.1']}
generate_multiple_graphic_loss([data12['error_test_m2'], data31['error_test_m2']], g_names, [data12['time'], data31['time']], save=False)

g_names = {'file_name': 'train_separate_m2', 'title': 'Train Error for target prediction', 'y_axis': 'Loss' , 'legend': ['Model 1.2', 'Model 3.1']}
generate_multiple_graphic_loss([data12['loss'], data31['loss']], g_names, [data12['time'], data31['time']], save=False)

"""

g_names = {'file_name': 'train_separate_m2', 'title': 'Train Loss Model 2', 'x_axis': 'Time [s]' , 'legend': ['Model 1.1', 'Model 1.2', 'Model 3.1']}
generate_multiple_graphic_loss([data11['loss'], data12['loss_2'], data31['loss_2']], g_names, [data11['time'],data12['time'], data31['time']], save=False)

g_names = {'file_name': 'train_separate_m2', 'title': 'Train Loss Model 2', 'x_axis': 'Epochs' , 'legend': ['Model 1.1', 'Model 1.2', 'Model 3.1']}
generate_multiple_graphic_loss([data11['loss'], data12['loss_2'], data31['loss_2']], g_names, save=False)

"""


#g_names = {'file_name': 'train_separate_m2', 'title': 'Train Loss Model 2', 'x_axis': 'Time [s]' , 'legend': ['Model 2.1', 'Model 2.2', 'Model 3.1']}
#generate_multiple_graphic_loss([data1['loss_m2'], data2['loss_m2'], data31['loss_2']], g_names, [data1['time_m2'],data2['time_m2'], data31['time']], save=False)

#g_names = {'file_name': 'train_separate_m2', 'title': 'Train Loss Model 2', 'x_axis': 'Epochs' , 'legend': ['Model 2.1', 'Model 2.2', 'Model 3.1']}
#generate_multiple_graphic_loss([data1['loss_m2'], data2['loss_m2'], data31['loss_2']], g_names, save=False)



#g_names = {'file_name': 'train_separate_m1_v1', 'title': 'Train Loss Model 1', 'x_axis': 'Epochs'}
#generate_graphic_loss(data['loss_m1'], g_names, save=False)

#g_names = {'file_name': 'train_separate_m2_v1', 'title': 'Train Loss Model 2', 'x_axis': 'time'}
#generate_graphic_loss(data['loss_m2'], g_names, data['time_m2'], save=False)

# print('Separate model - Test evaluation')
# errors, loss = test_model_separate(100, train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin, runs=5, epochs=50, version=1)
# g_names = {'file_name': 'test_separate_v1', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=True)


# EVALUATE SEPARATE MODEL WITH CROSS-ENTROPY AS LOSS (model 2.1)

#print('Separate model - Train evaluation - CROSS_ENTROPY')
#_, loss21 = test_model_separate(250, train_input, train_classes_one_hot, train_target_bin, runs=4, epochs=10, version=2, verbose=2)
#g_names = {'file_name': 'train_separate_v2', 'title': 'Train Loss'}
#generate_graphic_loss(loss, g_names, save=False)

# print('Separate model - Test evaluation')
# errors, loss = test_model_separate(250, train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin, runs=5, epochs=15, version=2)
# g_names = {'file_name': 'test_separate_v2', 'title': 'Train Loss'}##
# generate_graphic_loss(loss, g_names, save=True)

# EVALUATE 2 SEPARATE MODEL WITH CROSS-ENTROPY AS LOSS (model 2.2)

#print('Separate model - Train evaluation - CROSS_ENTROPY')
#_, loss22 = test_model_separate(250, train_input, train_classes_one_hot, train_target_bin, runs=4, epochs=10, version=3, verbose=2)
#g_names = {'file_name': 'train_separate_v2', 'title': 'Train Loss'}
#generate_graphic_loss(loss, g_names, save=False)

#generate_multiple_graphic_loss([loss21, loss22], g_names)

# print('Separate model - Test evaluation')
# errors, loss = test_model_separate(250, train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin, runs=5, epochs=15, version=2)
# g_names = {'file_name': 'test_separate_v2', 'title': 'Train Loss'}##
# generate_graphic_loss(loss, g_names, save=True)


# EVALUATE JOINT MODEL WITH INTERMEDIATE LOSS (model 3.1)

#print('Joint model - Train evaluation')
#errors, loss = test_model_joint(train_input, train_classes_one_hot, train_target_bin, epochs=100)
#g_names = {'file_name': 'train_joint', 'title': 'Train Loss'}
#generate_graphic_loss(loss, g_names, save=False)

# print('Joint model - Test evaluation')
# errors, loss = test_model_joint(train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin, epochs=100)
# g_names = {'file_name': 'test_joint', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=True)



if __name__ == '__main__':
    run()