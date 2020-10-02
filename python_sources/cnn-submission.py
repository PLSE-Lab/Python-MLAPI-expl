from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from skimage import img_as_ubyte
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

def NN():
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )
    return net1

def get_data():
    dataset = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    target = dataset[[0]].values.ravel()
    train = dataset.iloc[:,1:].values
    target = target.astype(np.uint8)
    train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
    return train,target,test

def preprocess(X,num):
    flat = []
    for i in range(0,num):
        thresh = threshold_otsu(X[i][0])
        X[i][0] = X[i][0] > thresh
        noisy_image = img_as_ubyte(X[i][0])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        X[i][0] = median(noisy_image, disk(1))
        flat.append(X[i][0].flatten())
    X_preprocessed = np.array(flat, 'float64')
    return X_preprocessed

train,target,test = get_data()
#train = preprocess(train,42000)
#test = preprocess(test,28000)

model = NN()
model.fit(train,target)

pred = model.predict(test)

np.savetxt('submission_cnn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')