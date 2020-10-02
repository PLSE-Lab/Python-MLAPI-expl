#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/sign_mnist_train.csv')
print('Dataframe Shape:', data.shape)


# In[ ]:


data.head()


# In[ ]:


x = data.iloc[:, 1:].values
print('Feature matrix:\n', x)
print('Shape of Feature matrix:', x.shape)


# In[ ]:


y = data.iloc[:, :1].values.flatten()
print('Labels:\n', y)
print('Shape of Labels:', y.shape)


# In[ ]:


def display(index):
    plt.imshow(x[index].reshape(28, 28), cmap = 'gray')
    plt.title(str(y[index]))
    plt.show()


# In[ ]:


display(77)


# In[ ]:


def one_hot_encode(y):
    return np.eye(25)[y]
y_encoded = one_hot_encode(y)
print('Shape of y after encoding:', y_encoded.shape)


# In[ ]:


y_encoded


# In[ ]:


# Hyperparameters
learning_rate = 0.001
batch_size = 128
epochs = 10000
display_step = 500


# In[ ]:


# Network Hyperparameters
n_inputs = 784
nh1 = 256
nh2 = 256
nh3 = 256
nh4 = 256
nh5 = 256
n_outputs = 25


# In[ ]:


X = tf.placeholder('float', [None, n_inputs])
Y = tf.placeholder('float', [None, n_outputs])


# In[ ]:


weights = {
    'w1' : tf.Variable(tf.random_normal([n_inputs, nh1])),
    'w2' : tf.Variable(tf.random_normal([nh1, nh2])),
    'w3' : tf.Variable(tf.random_normal([nh2, nh3])),
    'w4' : tf.Variable(tf.random_normal([nh3, nh4])),
    'w5' : tf.Variable(tf.random_normal([nh4, nh5])),
    'out_w' : tf.Variable(tf.random_normal([nh5, n_outputs]))
}


# In[ ]:


biases = {
    'b1' : tf.Variable(tf.random_normal([nh1])),
    'b2' : tf.Variable(tf.random_normal([nh2])),
    'b3' : tf.Variable(tf.random_normal([nh3])),
    'b4' : tf.Variable(tf.random_normal([nh4])),
    'b5' : tf.Variable(tf.random_normal([nh5])),
    'out_b' : tf.Variable(tf.random_normal([n_outputs]))
}


# In[ ]:


def neural_network(x, weights, biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['w2']), biases['b2']))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['w3']), biases['b3']))
    layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, weights['w4']), biases['b4']))
    layer5 = tf.nn.relu(tf.add(tf.matmul(layer4, weights['w5']), biases['b5']))
    layer_out = tf.matmul(layer5, weights['out_w']) + biases['out_b']
    return layer_out


# In[ ]:


logits = neural_network(X, weights, biases)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[ ]:


init = tf.global_variables_initializer()


# In[ ]:


def next_batch(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# In[ ]:


with tf.Session() as sess:
    
    sess.run(init)
    
    cost_hist, acc_hist = [], []
    
    for epoch in range(1, epochs + 1):
        batch_x, batch_y = next_batch(batch_size, x, y_encoded)
        
        sess.run(training_op, feed_dict = { X : batch_x, Y : batch_y })
        
        if epoch % display_step == 0:
            c, acc = sess.run([loss_op, accuracy], feed_dict = { X : batch_x, Y : batch_y })
            cost_hist.append(c)
            acc_hist.append(acc)
            print('Epoch ' + str(epoch) + ', Cost: ' + str(c) + ', Accuracy: ' + str(acc))
    
    W = sess.run(weights)
    B = sess.run(biases)
    print('Accuracy on train data: ' + str(sess.run(accuracy, feed_dict = { X : x, Y : y_encoded }) * 100) + ' %')
#     print('Accuracy on test data: ' + str(sess.run(accuracy, feed_dict = { X : x_test, Y : y_test }) * 100) + ' %')


# In[ ]:


plt.plot(list(range(len(cost_hist))), cost_hist)
plt.title("Change in cost")
plt.show()


# In[ ]:


plt.plot(list(range(len(acc_hist))), acc_hist)
plt.title("Change in accuracy")
plt.show()


# In[ ]:


def neural_network(x, weights, biases):
    layer1 = np.matmul(x, weights['w1']) + biases['b1']
    layer2 = np.matmul(layer1, weights['w2']) + biases['b2']
    layer3 = np.matmul(layer2, weights['w3']) + biases['b3']
    layer4 = np.matmul(layer3, weights['w4']) + biases['b4']
    layer5 = np.matmul(layer4, weights['w5']) + biases['b5']
    layer_out = np.matmul(layer5, weights['out_w']) + biases['out_b']
    return layer_out


# In[ ]:


def get_predictions(x, w, b):
    pred = neural_network(x, w, b)
    images, predictions = [], []
    for i in x:
        images.append(i.reshape(28, 28))
    for i in pred:
        predictions.append(list(i))
    predictions = [chr(int(i.index(max(i))) + ord('A')) for i in predictions]
    return (images, predictions)


# In[ ]:


images, preds = get_predictions(x, W, B)


# In[ ]:


plt.imshow(images[1])
plt.title(preds[1])
plt.show()


# In[ ]:


for key in W.keys():
    np.save(key, W[key])


# In[ ]:


for key in B.keys():
    np.save(key, B[key])

