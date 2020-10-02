#!/usr/bin/env python
# coding: utf-8

# # LeNet-5 CNN with TensorFlow:
# ** I am developing a series of kernels for different Deep Learning Models: **
# 
# * [L-Layered Neural Network from scratch](https://www.kaggle.com/curiousprogrammer/l-layered-neural-network-from-scratch)
# * [TensorFlow NN with Augmentation](https://www.kaggle.com/curiousprogrammer/digit-recognizer-tensorflow-nn-with-augmentation)
# * [Data Augmentation in Python, TF, Keras, Imgaug](https://www.kaggle.com/curiousprogrammer/data-augmentation-in-python-tf-keras-imgaug)
# * [Deep NN with Keras](https://www.kaggle.com/curiousprogrammer/deep-nn-with-keras-97-5) 
# * CNN with TensorFlow - This one
# * CNN with Keras
# * AutoEncoders with TensorFlow
# * AutoEncoders with Keras
# * GANs with TensorFlow
# * GANs with Keras

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load Data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


X_train = df_train.iloc[:, 1:]
Y_train = df_train.iloc[:, 0]


# In[ ]:


X_train.head()


# In[ ]:


Y_train.head()


# # Normalization

# In[ ]:


X_train = np.array(X_train)
Y_train = np.array(Y_train)


# In[ ]:


X_train = X_train/255.0


# In[ ]:


# dev-val split
X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.03, shuffle=True, random_state=2019)

#Reshape the arrays to match the input in tensorflow graph
X_dev = X_dev.reshape((X_dev.shape[0], 28, 28, 1))
X_val = X_val.reshape((X_val.shape[0], 28, 28, 1))


# # Plot digits

# In[ ]:


def plot_digits(X, Y):
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.tight_layout()
        plt.imshow(X[i].reshape((28, 28)), cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[ ]:


plot_digits(X_train[-20:], Y_train[-20:])


# # CNN Architecture
# 
# We will LeNet-5 CNN architeture to build our model.
# 
# ** LeNet - 5 Architecture: **
# 
# ![LeNet-5 Architecture](https://engmrk.com/wp-content/uploads/2018/09/LeNet_Original_Image.jpg)
# 
# ** Convolution Operation: **
# 
# ![Convolution Operation](https://www.researchgate.net/profile/Ihab_S_Mohamed/publication/324165524/figure/fig3/AS:611103423860736@1522709818959/An-example-of-convolution-operation-in-2D-2.png)
# 
# ### Input : Flattened 784px grayscale images, which can be represented as dimension (n, 28, 28, 1)
# ### Output: 0 - 9 
# 
# ### Let's decode the operations we will be performing in each layer 
# ** First Layer:  Convolutional Layer (CONV1): **
# * Parameters: Input (N) = 28, Padding (P) = 2, Filter (F) = 5 x 5, Stride (S) = 1
# * Conv Operation: ((N + 2P - F) / S) + 1 = ((28 + 4 - 5) / 1) + 1 = 28 x 28 
# * We will apply 6 filters / kernels so we will get a 28 x 28 x 6 dimensional output
# 
# ** Second Layer:  Average Pooling Layer (POOL1): **
# * Parameters: Input (N) = 28, Filter (F) = 2 x 2, Stride (S) = 2
# * AVG Pooling Operation: ((N + 2P -F) / S) + 1 = ((28 - 2) / 2) + 1 = 14 x 14
# * We will have a 14 x 14 x 6 dimensional output at the end of this pooling
# 
# ** Third Layer:  Convolutional Layer (CONV2): **
# * Parameters: Input (N) = 14, Filter (F) = 5 x 5, Stride (S) = 1
# * Conv Operation: ((N + 2P - F) / S) + 1 = ((14 - 5) / 1) + 1 = 10 x 10
# * We will apply 16 filters / kernels so we will get a 10 x 10 x 6 dimensional output 
# 
# ** Fourth Layer: Average Pooling Layer (POOL2): **
# * Parameters: Input (N) = 10, Filter (F) = 2 x 2, Stride (S) = 2
# * AVG Pooling Operation: ((N + 2P -F) / S) + 1 = ((10 - 2) / 2) + 1 = 5 x 5
# * We will have a 5 x 5 x 16 dimensional output at the end of this pooling
# 
# ** Fifth Layer: Fully Connected layer(FC1): **
# * Parameters: W: 400 * 120, b: 120
# * We will have an output of 120 x 1 dimension
# 
# ** Sixth Layer: Fully Connected layer(FC2): **
# * Parameters: W: 120 * 84, b: 84
# * We will have an output of 84 x 1 dimension
# 
# ** Seventh Layer: Output layer(Softmax): **
# * Parameters: W: 84 * 10, b: 10
# * We will get an output of 10 x 1 dimension
# 
# We will tweak the pooling layers from average to max and activation functions.

# # Input placeholder

# In[ ]:


x = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='Y')


# # Graph Creation - TF Layers

# In[ ]:


conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same', strides=1, activation='relu', name='CONV1')
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, name='POOL1')
conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, strides=1, activation='relu', name='CONV2')
pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, name='POOL2')
flatten1 = tf.layers.Flatten()(pool2)
fc1 = tf.layers.Dense(120, activation='relu')(flatten1)
fc2 = tf.layers.Dense(84, activation='relu')(fc1)
out = tf.layers.Dense(10, activation='softmax')(fc2)


# # Hyperparameters

# In[ ]:


batch_size = 100
learning_rate = 5e-4
epochs = 20


# # Cost, Accuracy, Optimizer

# In[ ]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out, name='cost'))
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
equal_pred = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
acc = tf.reduce_mean(tf.cast(equal_pred, tf.float32))


# # Initialise the variables

# In[ ]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# # Training the Model

# In[ ]:


T_dev = pd.get_dummies(Y_dev).values
T_val = pd.get_dummies(Y_val).values
for epoch in range(epochs):
    start_index = 0
    s = np.arange(X_dev.shape[0])
    np.random.shuffle(s)
    X_dev = X_dev[s, :]
    T_dev = T_dev[s]
    while start_index < X_dev.shape[0]:
        end_index = start_index + batch_size
        if end_index > X_dev.shape[0]:
            end_index = X_dev.shape[0]
        x_dev = X_dev[start_index:end_index, :]
        t_dev = T_dev[start_index:end_index]
        dev_cost, dev_acc, _ = sess.run([cost, acc, opt], feed_dict={x:x_dev, y:t_dev})
        start_index = end_index
    dev_cost, dev_acc = sess.run([cost, acc], feed_dict={x:X_dev, y:T_dev})
    val_cost, val_acc = sess.run([cost, acc], feed_dict={x:X_val, y:T_val})
    print('Epoch:{0} Cost:{1:5f} Acc:{2:.5f} Val_Cost:{3:5f} Val_Accuracy:{4:.5f}'.
          format(epoch+1, dev_cost, dev_acc, val_cost, val_acc))


# In[ ]:


X_test = pd.read_csv('../input/test.csv')
X_test = np.array(X_test)
X_test = X_test / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
T_test = sess.run([out], feed_dict={x:X_test})


# In[ ]:


Y_test = np.argmax(T_test[0], axis=1)
Y_test[:5]


# In[ ]:


df_out = pd.read_csv('../input/sample_submission.csv')
df_out['Label'] = Y_test
df_out.to_csv('out.csv', index=False)


# In[ ]:




