import numpy as np
import pandas as pd

import tensorflow as tf
import tflearn

from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import batch_normalization, local_response_normalization

data = []
labels = []

#df = pd.read_csv('../input/beacon_readings.csv')
#data = df.iloc[:, 0:3]
#labels = df.iloc[:, 3]

def get_train_data():
    with open("../input/beacon_readings.csv", "r") as fp:
        emp_data = fp.readlines()
        X = []
        Y = []
        cnt = 1
        for line in emp_data:
            line = line.strip()
            arr = line.split(',')
            Y.append([ float(arr[3]) ])
            X.append([ float(arr[0]), float(arr[1]), float(arr[2]) ])
            cnt += 1
            #X.append(int(arr[0]))
        return X, Y

data, labels = get_train_data()

np_data = np.array(data)
np_labels = np.array(labels)

mean_data = []
mean_labels = []

mean_data.append([ np.mean(np_data[0:39, 0]), np.mean(np_data[0:39, 1]), np.mean(np_data[0:39, 2]) ])
mean_labels.append([ labels[0][0] ])
mean_data.append([ np.mean(np_data[39:91, 0]), np.mean(np_data[39:91, 1]), np.mean(np_data[39:91, 2]) ])
mean_labels.append([ labels[39][0] ])
mean_data.append([ np.mean(np_data[91:147, 0]), np.mean(np_data[91:147, 1]), np.mean(np_data[91:147, 2]) ])
mean_labels.append([ labels[91][0] ])
mean_data.append([ np.mean(np_data[147:198, 0]), np.mean(np_data[147:198, 1]), np.mean(np_data[147:198, 2]) ])
mean_labels.append([ labels[147][0] ])
mean_data.append([ np.mean(np_data[198:250, 0]), np.mean(np_data[198:250, 1]), np.mean(np_data[198:250, 2]) ])
mean_labels.append([ labels[198][0] ])

"""data = [[0.], [1.]]
labels = [[1.], [0.]]

# Graph definition
with tf.Graph().as_default():
    g = tflearn.input_data(shape=[None, 1])
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(g, optimizer='sgd', learning_rate=2., loss='mean_square')
    
    # Model training
    m = tflearn.DNN(g)
    m.fit(data, labels, n_epoch=100, snapshot_epoch=False)
    # Test model
    print("Testing NOT operator")
    print("NOT 0:", m.predict([[0.]]))
    print("NOT 1:", m.predict([[1.]])) """

# Build neural network
net = tflearn.input_data(shape=[None, 3])
net = batch_normalization(net)
net = tflearn.fully_connected(net, 512, activation='relu', regularizer='L2')
#net = tflearn.dropout(net, 0.5)
net = batch_normalization(net)
net = tflearn.fully_connected(net, 256, activation='relu', regularizer='L2')
#net = tflearn.dropout(net, 0.5)
net = batch_normalization(net)
net = tflearn.fully_connected(net, 128, activation='relu', regularizer='L1')
#net = tflearn.dropout(net, 0.5)
net = batch_normalization(net)
net = tflearn.fully_connected(net, 64, activation='relu', regularizer='L1')
#net = tflearn.dropout(net, 0.5)
net = batch_normalization(net)
net = tflearn.fully_connected(net, 128, activation='linear', regularizer='L1')
#net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 1, activation='linear', regularizer='L1')
#net = tflearn.single_unit(net)
net = tflearn.regression(net, optimizer='sgd', loss='mean_square', learning_rate=0.01)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=100, batch_size=25, show_metric=False)

for i in range(len(labels)):
    print ("No. " + str(i) + " Predicted: " + str(model.predict([data[i]])) + " Target: " + str(labels[i]) + " Difference: " + str(model.predict([data[i]])[0][0] - labels[i][0]))
