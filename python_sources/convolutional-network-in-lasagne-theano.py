import sys, os
import theano
import theano.tensor as T
import numpy as np
import pandas as pd
import lasagne
from sklearn import cross_validation, metrics

#Load the training set and do normalization and re-shaping 
train = pd.read_csv('../input/train.csv').dropna().values
tr_X = train[:,1:].reshape(-1,1,28,28).astype(theano.config.floatX)
tr_X = tr_X/255
ty_Y = train[:,0]

#Split the training data into training and validation sets 
X_train,X_val,y_train,y_val = cross_validation.train_test_split(tr_X,ty_Y, test_size=0.1)

# Convolution Neural Network Model
# Build the convolutional network model 
#   Input           -> 1 x 28 x 28
#   Conv2D          -> 32 x 5 x 5
#   maxPool2D       -> 2x2
#   Dropout         -> p = 0.5
#   Conv2D          -> 32 x 5 x 5
#   maxPool2D       -> 2x2
#   Dropout         -> p = 0.5
#   Full Connected  -> 1024
#   Full Connected  -> 256
#   Full Connected  -> 10 (softmax)
#
#   cost function : categorical cross entropy
#   regularization: l2

def model():

    X = T.tensor4('X')
    y = T.ivector('y')

    net = lasagne.layers.InputLayer(shape=(None,1,28,28), input_var=X)
    net = lasagne.layers.Conv2DLayer(net,
                                    num_filters=32,
                                    filter_size=(5,5),
                                    nonlinearity = lasagne.nonlinearities.rectify,
                                    W=lasagne.init.GlorotUniform())
                                    
    net = lasagne.layers.MaxPool2DLayer(net, pool_size=(2,2))
    
    net = lasagne.layers.Conv2DLayer(lasagne.layers.DropoutLayer(net),
                                    num_filters=32,
                                    filter_size=(5,5),
                                    nonlinearity = lasagne.nonlinearities.rectify,
                                    W=lasagne.init.GlorotUniform())
                                    
    net = lasagne.layers.MaxPool2DLayer(net, pool_size=(2,2))
    
    net = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(net),
                                    num_units=1024,
                                    nonlinearity = lasagne.nonlinearities.rectify)
                                    
    net = lasagne.layers.DenseLayer(net,
                                    num_units=256,
                                    nonlinearity = lasagne.nonlinearities.rectify)
                                    
    net = lasagne.layers.DenseLayer(net,
                                    num_units=10,
                                    nonlinearity = lasagne.nonlinearities.softmax)

    params = lasagne.layers.get_all_params(net, trainable=True)
    prediction = lasagne.layers.get_output(net)
    cost = lasagne.objectives.categorical_crossentropy(prediction,y)
    cost = cost.mean() 
    cost += 0.0001 * lasagne.regularization.regularize_network_params(net,
                                            lasagne.regularization.l2)
    updates = lasagne.updates.adagrad(cost,params, learning_rate=0.01)
    train = theano.function([X,y],
                            [cost,prediction],
                            updates = updates,
                            allow_input_downcast=True)

    true_pred = lasagne.layers.get_output(net, deterministic=True)
    y_pred = T.argmax(true_pred, axis = 1)
    predict= theano.function([X], y_pred, allow_input_downcast=True)
    
    return train,predict

#Helper function to iterate over mini batches
def batch(X,y,n=128, randomize=True):
    l = len(X)
    if randomize:
        perm = np.random.permutation(l)
        Xrand = X[perm]
        yrand = y[perm]
    else:
        Xrand = X
        yrand = y
    for i in range(0,l,n):
        yield Xrand[i:min(i+n,l)],yrand[i:min(i+n,l)]

#create the network
fit,predict = model()

#training
for i in range(1):
    for X_batch,y_batch in batch(X_train,y_train):
        cost,pred = fit(X_batch,y_batch)
    if(i % 10 == 9):
        #validate score
        acc = metrics.accuracy_score(y_val, predict(X_val))
        print('Epoch {0}/{1} : cost : {2} Acc:{3}'.format(i,100,cost,acc))
        sys.stdout.flush()

#prediction
test = pd.read_csv('../input/test.csv').dropna().values
test = test.reshape(-1,1,28,28).astype(theano.config.floatX)
test = test/255
np.savetxt('submission.csv', np.c_[np.arange(1,len(test) + 1),predict(test)],
            header = 'ImageId,Label', delimiter=',', comments = '', fmt='%d')