#!/usr/bin/env python
# coding: utf-8

# In[181]:


nb_name = 'mnist-lenet-spatial-domain'


# In[182]:


#################### Output ####################
log_path_dev = f'log-{nb_name}-dev.csv'
log_path_train = f'log-{nb_name}-train.csv'
params_path = f'params-{nb_name}.csv'
submission_path = f'submisison.csv'


# ## Loading MNIST Dataset

# In[209]:


import os
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# In[184]:


def load_dataset(val_split = 0.25):
    np.random.seed(0)
    train_data = pd.read_csv('../input/train.csv')
    test_data = pd.read_csv("../input/test.csv")
    train_y = train_data['label'].values
    train_x = train_data.drop(columns=['label']).values
    train_x = train_x.reshape(-1,28,28)
    test_x = test_data.values
    test_x = test_x.reshape(-1,28,28)

    # Data Normalization
    mean, std = train_x.mean(), train_x.std()
    train_x = (train_x - mean)/std
    test_x = (test_x - mean)/std
    
    train_x = train_x.reshape((-1, 1, 28, 28))
    test_x = test_x.reshape((-1, 1, 28, 28))
    
    # Validation Set
    indices = np.arange(len(train_x))
    np.random.shuffle(indices)
    pivot = int(len(train_x) * (1 - val_split))
    train_x, val_x = train_x[indices[:pivot]], train_x[indices[pivot:]]
    train_y, val_y = train_y[indices[:pivot]], train_y[indices[pivot:]]
    
    return train_x, train_y, val_x, val_y, test_x


# In[185]:


train_x, train_y, val_x, val_y, test_x = load_dataset(1 - 28/42)
train_x.shape, val_x.shape, test_x.shape


# In[186]:


perm=[0, 3, 2, 1] # Back and forth between NHWC and NCHW
train_x, val_x, test_x = train_x.transpose(perm), val_x.transpose(perm), test_x.transpose(perm)
train_x.shape, val_x.shape, test_x.shape


# ## Model Architecture

# In[187]:


def conv_layer(name, x, k, f1, f2, s=1, padding='SAME'):
    with tf.variable_scope(name):
        value = tf.truncated_normal([k, k, f1, f2], stddev=1e-1)
        w = tf.get_variable('w', initializer=value)
        conv = tf.nn.conv2d(x, w, [1, 1, s, s], padding)

        value = tf.constant(1e-1, tf.float32, [f2])
        bias = tf.get_variable('bias', initializer=value)
        out = tf.nn.relu(tf.nn.bias_add(conv, bias))
        
        tf.summary.histogram('weights', w)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('activations', out)
        
        return out

def pool_layer(name, x, stride=2, padding='SAME'):
    with tf.variable_scope(name):
        param = [1, stride, stride, 1]
        x = tf.nn.max_pool(x, param, param, padding)
        return x

def conv_net(x, out=10, is_training=True):
    k, f1, f2, h1, h2 = 5, 6, 16, 120, 84 # LeNet
    
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
        # Input Channels     
        c = x.shape.as_list()[-1] # For NHWC

        x = conv_layer('Conv1', x, k, c, f1)
        x = pool_layer('Pool1', x)
        x = conv_layer('Conv2', x, k, f1, f2)
        x = pool_layer('Pool2', x)

        x = tf.contrib.layers.flatten(x, scope='Flatten')
        x = tf.contrib.layers.fully_connected(x, h1, tf.nn.relu, scope='Dense1')
        x = tf.contrib.layers.fully_connected(x, h2, tf.nn.relu, scope='Dense2')
        y = tf.contrib.layers.fully_connected(x, out, None, scope='Logits')
        return y


# ## Setting up Training Environment

# In[188]:


#################### Dataset Iteration ####################
def batch(x, y, batch_size=256):
    num_steps = len(x) // batch_size
    remainder = len(x) % batch_size
    samples = np.arange(len(x))
    np.random.shuffle(samples)
    
    for step in range(num_steps):
        a, b = step * batch_size, (step + 1) * batch_size
        yield x[samples[a:b]], y[samples[a:b]]

    '''
    a, b = num_steps * batch_size, num_steps * batch_size + remainder
    yield x[samples[a:b]], y[samples[a:b]]
    '''


# In[189]:


def get_trainable_params(params_path = None):
    params = list()
    columns = ['name', 'shape', 'params']
    for var in tf.trainable_variables():
        params.append([
            var.name,
            var.shape,
            np.prod(var.shape.as_list())
        ])
    return pd.DataFrame(params, columns=columns)


# # Development Phase
# ## Training Model
# ### Tweaks allowed based on validation feedback

# In[190]:


num_epochs = 100
val_step = int(num_epochs/10)
learning_rate = 1e-3


# In[191]:


tf.reset_default_graph()

image_shape = train_x.shape[1:]
    
x = tf.placeholder(tf.float32, shape=(None, *image_shape))
y = tf.placeholder(tf.int64, shape=(None,))

# Building Model
logits = conv_net(x)

with tf.name_scope('loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(y, logits)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('accuracy'):
    prediction = tf.argmax(logits, 1)
    true_pos = tf.equal(y, prediction)
    accuracy = tf.reduce_mean(tf.cast(true_pos, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


# In[192]:


sess = tf.Session()


# In[193]:


get_ipython().run_cell_magic('time', '', "\n# Initializing the variables\nsess.run(tf.global_variables_initializer())\n\nmerged_summary = tf.summary.merge_all()\nwriter = tf.summary.FileWriter('./summary')\nwriter.add_graph(sess.graph)\n\nlog = list()\nprint('Training Model')\nfor epoch in range(num_epochs):\n    # Model Training\n    tic = time.perf_counter()\n\n    batches = list(batch(train_x, train_y))\n    for i, (batch_x, batch_y) in enumerate(batches):\n        if i % int(num_epochs * 1.5) == 0:\n            s = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})\n            writer.add_summary(s, epoch*len(batches) + i)\n        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})\n\n    tac = time.perf_counter()\n    tictac = tac-tic\n\n    # Model Validation\n    if (epoch+1)%val_step == 0:\n        train_loss, train_acc, val_loss, val_acc = [], [], [], []\n\n        # Evaluation on Training Set\n        for batch_x, batch_y in batch(train_x, train_y):\n            loss_, acc_ = sess.run((loss, accuracy), feed_dict={x: batch_x, y: batch_y})\n            train_loss.append(loss_)\n            train_acc.append(acc_)\n\n        # Evaluation on Validation Set\n        for batch_x, batch_y in batch(val_x, val_y):\n            loss_, acc_ = sess.run((loss, accuracy), feed_dict={x: batch_x, y: batch_y})\n            val_loss.append(loss_)\n            val_acc.append(acc_)\n\n        train_loss = np.array(train_loss).sum()\n        train_acc = np.array(train_acc).mean() * 100\n        \n        val_loss = np.array(val_loss).sum()\n        val_acc = np.array(val_acc).mean() * 100\n\n        params = [epoch+1, tictac, train_loss, train_acc, val_loss, val_acc]\n        msg = 'epoch: {}'\n        msg += ' time: {:.3f}s train_loss: {:.3f} train_acc: {:.3f}'\n        msg += ' val_loss: {:.3f} val_acc: {:.3f}'\n        print(msg.format(*params))\n        log.append(params)")


# In[194]:


df_params = get_trainable_params()
df_params.to_csv(params_path)
df_params


# In[195]:


df_params['params'].sum()


# In[196]:


cols = ['epoch', 'time', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
df_log_dev = pd.DataFrame(log, columns=cols)
df_log_dev = df_log_dev.set_index('epoch')
df_log_dev.to_csv(log_path_dev)
df_log_dev


# In[197]:


df_log_dev.mean()


# In[219]:


fig, ax_ = plt.subplots(2,1, figsize=(9,7))

df_error_rate = df_log_dev[['train_loss', 'val_loss']]
ax = df_error_rate.plot(title='Overfit Analysis', marker='.', ax=ax_[0])
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss');

df_error_rate = df_log_dev[['train_acc', 'val_acc']]
ax = df_error_rate.plot(marker='.', ax=ax_[1])
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy');


# # Retraining Model

# In[199]:


data_x = np.concatenate((train_x, val_x))
data_y = np.concatenate((train_y, val_y))


# In[201]:


get_ipython().run_cell_magic('time', '', "\nlog = list()\nprint('Training Model')\nfor epoch in range(num_epochs):\n    # Model Training\n    tic = time.perf_counter()\n\n    batches = list(batch(data_x, data_y))\n    for i, (batch_x, batch_y) in enumerate(batches):\n        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})\n\n    tac = time.perf_counter()\n    tictac = tac-tic\n\n    # Model Validation\n    if (epoch+1)%val_step == 0:\n        train_loss, train_acc = list(), list()\n\n        # Evaluation on Training Set\n        for batch_x, batch_y in batch(train_x, train_y):\n            loss_, acc_ = sess.run((loss, accuracy), feed_dict={x: batch_x, y: batch_y})\n            train_loss.append(loss_)\n            train_acc.append(acc_)\n\n        train_loss = np.array(train_loss).sum()\n        train_acc = np.array(train_acc).mean() * 100\n\n        params = [epoch+1, tictac, train_loss, train_acc]\n        msg = 'epoch: {}'\n        msg += ' time: {:.3f}s train_loss: {:.3f} train_acc: {:.3f}'\n        print(msg.format(*params))\n        log.append(params)")


# In[202]:


cols = ['epoch', 'time', 'train_loss', 'train_acc']
df_log_train = pd.DataFrame(log, columns=cols)
df_log_train = df_log_train.set_index('epoch')
df_log_train.to_csv(log_path_train)
df_log_train


# In[220]:


fig, ax_ = plt.subplots(2,1, figsize=(9,7))

df_error_rate = df_log_train[['train_loss']]
ax = df_error_rate.plot(title='Overfit Analysis', marker='.', ax=ax_[0])
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss');

df_error_rate = df_log_train[['train_acc']]
ax = df_error_rate.plot(marker='.', ax=ax_[1])
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy');


# In[204]:


test_y = sess.run(prediction, feed_dict={x: test_x})


# In[205]:


output = pd.DataFrame(test_y, columns=['Label'])
output.index = np.arange(1, len(output) + 1)
output.index.names = ['ImageId']
output.to_csv(submission_path)

