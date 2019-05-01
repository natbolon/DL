from dlc_practical_prologue import generate_pair_sets
from generate_data import binarize, shuffle, normalize_data, to_one_hot
from torch.autograd import Variable

from train import test_model_separate, test_model_joint, test_model_fc
from graphics import generate_graphic_loss


print('Generate data')
train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

# convert to binary target
train_target_bin, test_target_bin = binarize(train_target), binarize(test_target)

# shuffle train set
train_input, train_classes, train_target_bin = shuffle(train_input, train_classes, train_target_bin)

# reshape
train_input = train_input.reshape((train_input.size(0) * 2, 1, train_input.size(2), train_input.size(3)))
test_input = test_input.reshape((test_input.size(0) * 2, 1, test_input.size(2), test_input.size(3)))

# normalize
normalize_data(train_input, test_input)

# convert to one hot encoding
train_classes_one_hot = to_one_hot(train_classes.view(train_classes.size(0) * 2, -1))
test_classes_one_hot = to_one_hot(test_classes.view(test_classes.size(0) * 2, -1))

print('Generate Variables')
train_input, train_target_bin, train_classes_one_hot, train_classes = Variable(train_input), Variable(train_target_bin), Variable(train_classes_one_hot), Variable(train_classes)
test_input, test_target_bin, test_classes_one_hot, test_classes = Variable(test_input), Variable(test_target_bin), Variable(test_classes_one_hot), Variable(test_classes)

print('FC model - Train evaluation')
_, loss = test_model_fc(train_input, train_classes_one_hot, train_target_bin, runs=10, epochs=60, l_rate=0.1)
g_names = {'file_name': 'train_fc_model', 'title': 'Train Loss'}
generate_graphic_loss(loss, g_names, save=False)

#print('FC model - Train evaluation')
#_, loss = test_model_fc(train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin,  runs=10, epochs=40)
#g_names = {'file_name': 'train_fc_model', 'title': 'Train Loss'}
#generate_graphic_loss(loss, g_names, save=False)



# print('Separate model - Train evaluation')
# _, loss = test_model_separate(100, train_input, train_classes_one_hot, train_target_bin, runs=3, epochs=25, version=1)
# g_names = {'file_name': 'train_separate_v1', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=False)


# print('Separate model - Test evaluation')
# errors, loss = test_model_separate(100, train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin, runs=5, epochs=50, version=1)
# g_names = {'file_name': 'test_separate_v1', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=True)

# print('Separate model - Train evaluation')
# _, loss = test_model_separate(250, train_input, train_classes_one_hot, train_target_bin, runs=5, epochs=50, version=2)
# g_names = {'file_name': 'train_separate_v2', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=False)


# print('Separate model - Test evaluation')
# errors, loss = test_model_separate(250, train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin, runs=5, epochs=15, version=2)
# g_names = {'file_name': 'test_separate_v2', 'title': 'Train Loss'}##
# generate_graphic_loss(loss, g_names, save=True)

# print('Joint model - Train evaluation')
# errors, loss = test_model_joint(train_input, train_classes_one_hot, train_target_bin, epochs=100)
# g_names = {'file_name': 'train_joint', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=False)
#
#
# print('Joint model - Test evaluation')
# errors, loss = test_model_joint(train_input, train_classes_one_hot, train_target_bin, test_input, test_classes_one_hot, test_target_bin, epochs=100)
# g_names = {'file_name': 'test_joint', 'title': 'Train Loss'}
# generate_graphic_loss(loss, g_names, save=True)



if __name__ == '__main__':
    run()