import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.cross_validation import train_test_split

train = pd.read_csv("../input/train.csv").values #warning!
test  = pd.read_csv("../input/test.csv").values
print(train.shape)
img_rows, img_cols = 28, 28

X_train = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype(float)
X_train /= 255.0
y_train = train[:, 0].astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=42)

print(X_train.shape)
print(X_test.shape)
import lasagne
import theano
import theano.tensor as T

def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=600,
                                   nonlinearity=lasagne.nonlinearities.rectify,
                                   W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.4)

    l_hid2 = lasagne.layers.DenseLayer( l_hid1_drop, num_units=600,
                                   nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.4)
    l_out = lasagne.layers.DenseLayer( l_hid2_drop, num_units=10,
                                      nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),input_var=input_var)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    
    #network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(5, 5),
    #                                     nonlinearity=lasagne.nonlinearities.rectify)
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=200,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=10,nonlinearity=lasagne.nonlinearities.softmax)

    return network

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
# Create neural network model
network = build_cnn(input_var)

'''The first step generates a Theano expression for the network output given the input variable linked to the network’s input
layer(s).
The second step defines a Theano expression for the categorical cross-entropy loss between said network output and
the targets. Finally, as we need a scalar loss, we simply take the mean over the mini-batch.
Depending on the problem you aresolving, you will need different loss functions, see lasagne.objectives for more.'''

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

'''Having the model and the loss function defined, we create update expressions for training the network.
An update expression describes how to change the trainable parameters of the network at each presented mini-batch.
We will use Stochastic Gradient Descent (SGD) with Nesterov momentum here, '''

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

# This is the function we'll use to compute the network's output given an input
# (e.g., for computing accuracy). Again, we don't want to apply dropout here
# so we set the deterministic kwarg to True.
get_output = theano.function([input_var], lasagne.layers.get_output(network, deterministic=True))
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

num_epochs = 20
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 800, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 800, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
        
    # print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))