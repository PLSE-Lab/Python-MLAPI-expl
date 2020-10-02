#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import tflearn

import pickle, gzip


# In[ ]:


pkl = gzip.open('../input/mnist.pkl.gz','rb')
data = pickle.load(pkl, encoding='latin1')
len(data)


# In[ ]:


(trainX, trainY), (testX, testY), (validX, validY) = data
print(trainX.shape)
print(testX.shape)
print(validX.shape)


# In[ ]:


def one_hot_encode(x):
    labels = []
    for label in x:
        one_hot = np.array([int(i == label) for i in range(10)])
        labels.append(one_hot)
    return np.array(labels)


# In[ ]:


trainY = one_hot_encode(trainY)
testY = one_hot_encode(testY)
validY = one_hot_encode(validY)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def show_digit(index):
    label = trainY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = trainX[index].reshape([28,28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()
    
show_digit(41)


# In[ ]:


def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    # Input Layer
    net = tflearn.input_data([None, 784])
    
    # Hidden layers
    net = tflearn.fully_connected(net, 300, activation='ReLU')
    net = tflearn.fully_connected(net, 100, activation='ReLU')
    
    # output layer
    net = tflearn.fully_connected(net, 10,  activation='softmax')
    
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.05, loss='categorical_crossentropy')
    
    # This model assumes that your network is named "net"    
    model = tflearn.DNN(net)
    return model


# In[ ]:


model = build_model()


# In[ ]:


model.fit(trainX, trainY, validation_set=0.1, batch_size=200, show_metric=False, n_epoch=25)


# In[ ]:


predictions = np.array(model.predict(testX)).argmax(axis=1)
actual = testY.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)

# Print out the result
print("Test accuracy: ", test_accuracy)


# In[ ]:


import pandas as pd

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("results.csv", index=False, header=True)


# In[ ]:




