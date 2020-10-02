
# Generate output files with write_csv(), plot() or ggplot()
# Any files you write to the current directory get shown as outputs
import numpy as np
import pandas as pd
import lasagne

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

training_data = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv").values

target = training_data[[0]].values.ravel()
train = training_data.iloc[:,1:].values

target = target.astype(np.uint8)
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)

def CNN2(n_epochs):
    net2 = NeuralNet(
            layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dense', layers.DenseLayer),                
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape=( None , 1, 28, 28),
        conv1_num_filters=32,
        conv1_filter_size=(5, 5),
        conv1_nonlinearity=lasagne.nonlinearities.rectify,
        conv1_W=lasagne.init.GlorotUniform(),
        pool1_pool_size=(2, 2),
        conv2_num_filters=32,
        conv2_filter_size=(5, 5),
        conv2_nonlinearity=lasagne.nonlinearities.rectify,
        pool2_pool_size=(2, 2),
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        dropout2_p=0.5,
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,
        update=nesterov_momentum,
        update_learning_rate=0.001,
        update_momentum=0.9,
        max_epochs=n_epochs,
        verbose=1
        )
    return net2

cnn2 = CNN2(5).fit(train, target)

preds = cnn2.predict(test)

submission = pd.DataFrame({
    "ImageId": range(1, len(test) + 1),
    "Label": preds
    })

submission.to_csv("kaggle-digit-classifier-2016-08-05.csv", index=False)
