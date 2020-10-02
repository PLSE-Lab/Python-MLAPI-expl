#!/usr/bin/env python
# coding: utf-8

# # Handwritten Digit Recognition
# 
# In this tutorial, we'll give you a step by step walk-through of how to build a hand-written digit classifier using the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. For someone new to deep learning, this exercise is arguably the "Hello World" equivalent.
# 
# MNIST is a widely used dataset for the hand-written digit classification task. It consists of 70,000 labeled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). The task at hand is to train a model using the 60,000 training images and subsequently test its classification accuracy on the 10,000 test images.
# 
# ![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/mnist.png)
# 
# **Figure 1:** Sample images from the MNIST dataset.

# In[ ]:


import mxnet as mx
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
path = '../input/'


# In[ ]:


# Fix the seed
mx.random.seed(7)

# Set the compute context, GPU is available otherwise CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()


# In[ ]:


print("model extract train init")
df_train = pd.read_csv(path + 'train.csv')
y = (np.array(df_train['label'].values.tolist()).astype(np.int)).copy()
df_train = df_train.drop(columns=['label'])
X = (np.array(df_train.values.tolist()).astype(np.float)).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("model extract train end")


# In[ ]:


print("model extract test init")
df_test_final = pd.read_csv(path + 'test.csv')
X_test_final = np.array(df_test_final.values.tolist()).astype(np.float)
print("model extract train end")


# In[ ]:


def reshare_array(array, dim):
    return np.reshape(array, (-1, 1, dim, dim))


# In[ ]:


X_train = reshare_array(X_train, 28)
X_test = reshare_array(X_test, 28)
X_test_final = reshare_array(X_test_final, 28)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


batch_size = 100
train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size)
val_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)


# ### Convolutional Neural Network
# 
# Earlier, we briefly touched on a drawback of MLP when we said we need to discard the input image's original shape and flatten it as a vector before we can feed it as input to the MLP's first fully connected layer. Turns out this is an important issue because we don't take advantage of the fact that pixels in the image have natural spatial correlation along the horizontal and vertical axes. A convolutional neural network (CNN) aims to address this problem by using a more structured weight representation. Instead of flattening the image and doing a simple matrix-matrix multiplication, it employs one or more convolutional layers that each performs a 2-D convolution on the input image.
# 
# A single convolution layer consists of one or more filters that each play the role of a feature detector. During training, a CNN learns appropriate representations (parameters) for these filters. Similar to MLP, the output from the convolutional layer is transformed by applying a non-linearity. Besides the convolutional layer, another key aspect of a CNN is the pooling layer. A pooling layer serves to make the CNN translation invariant: a digit remains the same even when it is shifted left/right/up/down by a few pixels. A pooling layer reduces a *n x m* patch into a single value to make the network less sensitive to the spatial location. Pooling layer is always included after each conv (+ activation) layer in the CNN.
# 
# The following source code defines a convolutional neural network architecture called LeNet. LeNet is a popular network known to work well on digit classification tasks. We will use a slightly different version from the original LeNet implementation, replacing the sigmoid activations with tanh activations for the neurons

# In[ ]:


data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')


# ![png](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/conv_mnist.png)
# 
# **Figure 3:** First conv + pooling layer in LeNet.
# 
# Now we train LeNet with the same hyper-parameters as before. Note that, if a GPU is available, we recommend using it. This greatly speeds up computation given that LeNet is more complex and compute-intensive than the previous multilayer perceptron. To do so, we only need to change `mx.cpu()` to `mx.gpu()` and MXNet takes care of the rest. Just like before, we'll stop training after 10 epochs.

# In[ ]:


lenet_model = mx.mod.Module(symbol=lenet, context=ctx)


# In[ ]:


# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.005},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=150)


# ### Prediction
# 
# Finally, we'll use the trained LeNet model to generate predictions for the test data.

# In[ ]:


test_iter = mx.io.NDArrayIter(X_test_final, None, batch_size)
prob = lenet_model.predict(test_iter)

y = []
for p in list(prob):
   y.append(list(p).index(np.max(p)))


# In[ ]:


test_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
lenet_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]


# In[ ]:


df = pd.DataFrame({'ImageId': [x for x in range(1, len(y) + 1)], 'Label': y})
df.to_csv('submission.csv', index=False)


# If all went well, we should see a higher accuracy metric for predictions made using LeNet. With CNN we should be able to correctly predict around 98% of all test images.
# 
# ## Summary
# 
# In this tutorial, we have learned how to use MXNet, based on [MXNet Examples](https://mxnet.incubator.apache.org/versions/master/tutorials/python/mnist.html).

# In[ ]:




