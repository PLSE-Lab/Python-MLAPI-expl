

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import lasagne
from lasagne import layers

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion


# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../input/test.csv").values

print(dataset.head())

train = np.array(train).astype(np.uint8)
target = np.array(target).astype(np.uint8)
test = np.array(test).astype(np.uint8)

# apply some very simple normalization to the data
#train -= train.mean()
#train /= train.std()

# For convolutional layers, the default shape of data is bc01,
# i.e. batch size x color channels x image dimension 1 x image dimension 2.
# Therefore, we reshape the X data to -1, 1, 28, 28.
train = train.reshape(
    -1,  # number of samples, -1 makes it so that this number is determined automatically
    1,   # 1 color channel, since images are only black and white
    28,  # first image dimension (vertical)
    28,  # second image dimension (horizontal)
)

test = test.reshape(
    -1,  # number of samples, -1 makes it so that this number is determined automatically
    1,   # 1 color channel, since images are only black and white
    28,  # first image dimension (vertical)
    28,  # second image dimension (horizontal)
)

figs, axes = plt.subplots(4, 4, figsize=(6, 6))
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(-train[i + 4 * j].reshape(28, 28), cmap='gray', interpolation='none')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        axes[i, j].set_title("Label: {}".format(target[i + 4 * j]))
        axes[i, j].axis('off')

def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses


def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
        ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 28, 28),
    conv1_num_filters=7, 
    conv1_filter_size=(3, 3), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=12, 
    conv2_filter_size=(2, 2),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
        
    hidden3_num_units=1000,
    output_num_units=10, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.0001,
    update_momentum=0.9,
    #objective=regularization_objective,
    #objective_lambda2=0.0025,    
    max_epochs=n_epochs,
    verbose=1,
    )
    return net1
cnn = CNN(15).fit(train,target) # train the CNN model for 15 epochs
#plot_loss(cnn)
# We can further have a look at the weights learned by the net.
#plot_conv_weights(cnn.layers_[1], figsize=(4, 4))

#x = train[0:1]

# To see through the "eyes" of the net, we can plot the activities produced by different layers
#plot_conv_activity(cnn.layers_[1], x)

# Plot occlusion images
# A possibility to check if the net, for instance, overfits or learns important 
# features is to occlude part of the image. Then we can check whether the net 
# still makes correct predictions
#plot_occlusion(cnn, train[:5], target[:5])
# use the NN model to classify test data
pred = cnn.predict(test)

# save results
np.savetxt('submission_cnn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

