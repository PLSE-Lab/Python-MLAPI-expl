#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import imgaug.augmenters as iaa

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load input data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


X_train = df_train.iloc[:, 1:]
Y_train = df_train.iloc[:, 0]


# In[ ]:


X_train.head()


# In[ ]:


X_train.shape


# In[ ]:


Y_train.head()


# In[ ]:


X_train = np.array(X_train)
Y_train = np.array(Y_train)


# # Plot images

# In[ ]:


def plot_digits(X, y, dim):
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(X[i].reshape((dim, dim)), interpolation='none', cmap='gray')
        plt.title("Digit: {}".format(y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[ ]:


plot_digits(X_train, Y_train, 28)


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 8))
sns.countplot(Y_train)
ax.set_title("Distribution of Digits", fontsize=12)
ax.set_xlabel("Digits", fontsize=10)
ax.set_ylabel('Count', fontsize=10)
plt.show()


# We have almost equal distribution of digits.

# # Normalize Data

# In[ ]:


X_train = X_train/255.0


# In[ ]:


plot_digits(X_train, Y_train, 28)


# # Train Test Split

# In[ ]:


X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.03, shuffle=True,                                               random_state=2019)


# In[ ]:


#Encode Y
def yEncode(Y):
    #T = np.zeros((Y.size, len(set(Y))))
    #T[np.arange(Y.size), Y] = 1
    T = pd.get_dummies(Y).values
    return T

T_dev = yEncode(Y_dev)
T_val = yEncode(Y_val)


# # Augmentation Implementation

# In[ ]:


class ImageAugmenter:
    def __init__(self):
        self.name = 'ImageAugmenter'
        
    def reshape_images(self, img_arr, shape):
        return img_arr.reshape(shape)
    
    def transform_images(self, seq, img_arr, shape):
        X_img = self.reshape_images(img_arr, (img_arr.shape[0], shape, shape))
        X_aug = seq.augment_images(X_img)
        X_aug = self.reshape_images(X_aug, (img_arr.shape[0], shape*shape))
        return X_aug
        
    def fliplr(self, X, shape):
        seq = iaa.Sequential([
            iaa.Fliplr(1)
        ])
        return self.transform_images(seq, X, shape)
    
    def flipud(self, X, shape):
        seq = iaa.Sequential([
            iaa.Flipud(1)
        ])
        return self.transform_images(seq, X, shape)
    
    def scale(self, X, shape):
        seq = iaa.Sequential([
            iaa.Affine(
                scale={"x":(0.5, 1.5), "y":(0.5, 1.5)}
            )
        ])
        return self.transform_images(seq, X, shape)
    
    def translate(self, X, shape):
        seq = iaa.Sequential([
            iaa.Affine(
                translate_percent={"x":(-0.2, 0.2), "y":(-0.2, 0.2)}
            )
        ])
        return self.transform_images(seq, X, shape)
    
    def rotate(self, X, shape):
        seq = iaa.Sequential([
            iaa.Affine(
                rotate=(-45, 45)
            )
        ])
        return self.transform_images(seq, X, shape)
    
    def shear(self, X, shape):
        seq = iaa.Sequential([
            iaa.Affine(
                shear=(-10, 10)
            )
        ])
        return self.transform_images(seq, X, shape)
    
    def compose(self, X, shape):
        seq = iaa.Sequential([
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
            ),
            iaa.Pepper(1e-5)
        ])
        return self.transform_images(seq, X, shape)
    
    def augment(self, count, X, Y):
        X_out = np.copy(X)
        for i in range(count):
            #X_scale = self.scale(X, 28)
            #X_rotate = self.rotate(X, 28)
            #X_trans = self.translate(X, 28)
            #X_shear = self.shear(X, 28)
            X_compose = self.compose(X, 28)
            #X_out = np.concatenate((X_out, X_scale), axis = 0)
            #X_out = np.concatenate((X_out, X_rotate), axis = 0)
            #X_out = np.concatenate((X_out, X_trans), axis = 0)
            #X_out = np.concatenate((X_out, X_shear), axis = 0)
            X_out = np.concatenate((X_out, X_compose), axis = 0)
        Y = np.repeat(Y, (count*1)+1, axis = 0)
        return X_out, Y


# # Placeholders for inputs and dropouts

# In[ ]:


X = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')


# In[ ]:


keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)


# # Initialise Parameters

# In[ ]:


def get_W(name, values):
    W = tf.Variable(values, name=name)
    return W


# In[ ]:


def get_b(name, values):
    b = tf.Variable(values, name=name)
    return b


# # Graph Functions

# In[ ]:


def linear_forward(X, W, b):
    z = tf.matmul(X, W) + b
    return z


# In[ ]:


def activation_forward(X, W, b, activation):
    z = linear_forward(X, W, b)
    if activation == 'sigmoid':
        return tf.nn.sigmoid(z)
    elif activation == 'softmax':
        return tf.nn.softmax(z)
    elif activation == 'tanh':
        return tf.nn.tanh(z)
    elif activation == 'relu':
        return tf.nn.relu(z)


# In[ ]:


def fc_layer(X, n_units, name, activation):
    W = get_W(name, tf.truncated_normal([X.shape[1].value, n_units], stddev=0.1))
    b = get_b(name, tf.zeros([1, n_units]))
    return activation_forward(X, W, b, activation)


# # Network Architecture

# In[ ]:


dropout_x = tf.nn.dropout(X, keep_prob=keep_prob_1)
fc1 = fc_layer(dropout_x, 324, 'fc1', 'relu')
dropout_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob_2)
fc2 = fc_layer(dropout_fc1, 100, 'fc2', 'relu')
dropout_fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob_2)
fc3 = fc_layer(dropout_fc2, 100, 'fc3', 'relu')
dropout_fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob_2)
fc4 = fc_layer(dropout_fc3, 100, 'fc4', 'relu')
dropout_fc4 = tf.nn.dropout(fc4, keep_prob=keep_prob_2)
fc5 = fc_layer(dropout_fc4, 225, 'fc5', 'relu')
dropout_fc5 = tf.nn.dropout(fc5, keep_prob=keep_prob_2)
out = fc_layer(dropout_fc5, 10, 'out', 'softmax')


# # Hyperparameters

# In[ ]:


epoch = 20
learning_rate = 5e-4
batch_size = 100


# # Cost, Accuracy and Optimizer

# In[ ]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out))


# In[ ]:


print(learning_rate)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# In[ ]:


equal_pred = tf.equal(tf.argmax(y,1), tf.argmax(out,1))
acc = tf.reduce_mean(tf.cast(equal_pred, tf.float32))


# # Initialise all variables

# In[ ]:


init = tf.global_variables_initializer()


# # Training the model

# In[ ]:


#Create session and initialise global variables
sess = tf.Session()
sess.run(init)
augmenter = ImageAugmenter()

for i in range(epoch):
    start_index = 0
    s = np.arange(X_dev.shape[0])
    np.random.shuffle(s)
    X_dev = X_dev[s,:]
    T_dev = T_dev[s]
    while start_index < X_dev.shape[0]:
        end_index = start_index + batch_size
        if end_index > X_dev.shape[0]:
            end_index = X_dev.shape[0]
        x_dev = X_dev[start_index:end_index, :]
        y_dev = T_dev[start_index:end_index]
        #x_dev, y_dev = augmenter.augment(1, x_dev, y_dev)
        #print(x_dev.shape, y_dev.shape)
        cost, _, accuracy, pred, fc = sess.run([loss, train, acc, out, fc2],                                     feed_dict={X: x_dev, y:y_dev, keep_prob_1:1, keep_prob_2:1})
        start_index = end_index
    t_cost, t_acc = sess.run([loss, acc],                                        feed_dict={X:X_dev, y:T_dev, keep_prob_1:1, keep_prob_2:1})
    v_cost, v_acc = sess.run([loss, acc],                                        feed_dict={X:X_val, y:T_val, keep_prob_1:1, keep_prob_2:1})
    X_grad, fc1_grad, fc2_grad, fc3_grad, fc4_grad, fc5_grad =                     sess.run([X, fc1, fc2, fc3, fc4, fc5],                         feed_dict={X:X_dev, y:T_dev, keep_prob_1:1, keep_prob_2:1})
    Y_dev = np.argmax(T_dev, axis=1)
    print("Epoch:", (i+1), "cost =", "{:.5f}".format(t_cost), "acc =", "{:.5f}".format(t_acc),              "val_cost = {:.5f}".format(v_cost), "val_acc = {:.5f}".format(v_acc))


# In[ ]:


plt.imshow(X_dev[2].reshape((28,28))), np.argmax(T_dev[2])


# In[ ]:


X_test = pd.read_csv('../input/test.csv')


# In[ ]:


X_test = np.array(X_test)


# In[ ]:


out_test = sess.run([out], feed_dict={X:X_test, keep_prob_1:1, keep_prob_2:1})


# In[ ]:


P_test = np.argmax(out_test[0], axis=1)


# In[ ]:


df_out = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df_out['Label'] = P_test


# In[ ]:


df_out.to_csv('out.csv', index=False)

