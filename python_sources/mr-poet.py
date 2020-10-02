#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import random

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/all.csv')


# In[3]:


content = df['content'].tolist()[:3]


# ## Convert sentence into list of words

# In[4]:


def sent_to_words(content):
    return [np.array([x.split() for x in poem.split()]) for poem in content]


# In[5]:


poems = sent_to_words(content)


# ## Convert char to number mapping

# In[6]:


def build_dict(poems):
    dictionary = {}
    rev_dict = {}
    count = 0
    for content in poems:
        for i in content:
            if i[0] in dictionary:
                pass
            else:
                dictionary[i[0]] = count
                count += 1
    rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, rev_dict


# In[7]:


dictionary, rev_dict = build_dict(poems)


# ## LSTM time

# In[8]:


import tensorflow as tf
from tensorflow.contrib import rnn


# In[9]:


vocab_size = len(dictionary)


# In[10]:


# Parameters
learning_rate = 0.0001
training_iters = 1600
display_step = 200
n_input = 9


# In[11]:


# number of units in RNN cell
n_hidden = 512


# In[12]:


# tf Graph input
tf.device("/device:GPU:0")
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])


# In[13]:


# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


# In[14]:


def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# In[15]:


pred = RNN(x, weights, biases)


# In[16]:


# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[17]:


# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[18]:


saver = tf.train.Saver()
init = tf.global_variables_initializer()


# In[19]:


df_train = sent_to_words(content)


# In[20]:


j = 0
for i in df_train:
    if i.shape[0] <= n_input:
        df_train = np.delete(df_train, (j), axis = 0)
        j -= 1
    j += 1


# In[22]:


with tf.Session() as session:
    session.run(init)
    step = 0
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    while step < training_iters:
        acc_total = 0
        loss_total = 0
        j = 0
        for training_data in df_train:
            m = training_data.shape[0]
            windows = m - n_input
            acc_win = 0
            for window in range(windows):
                batch_x = training_data[window : window + n_input]
                batch_y = training_data[window + n_input]
                symbols_in_keys = [dictionary[i[0]] for i in batch_x]
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        
                symbols_out_onehot = np.zeros([vocab_size], dtype=float)
                symbols_out_onehot[dictionary[batch_y[0]]] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

                _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                loss_total += loss
                acc_win += acc
            acc_total += float(acc_win) / m
        acc_total /= len(df_train)
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " +                   "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " +                   "{:.2f}%".format(100*acc_total))
        step += 1
    print("Optimization Finished!")
    save_path = saver.save(session, "../working/model.ckpt")
    print("Model saved in path: %s" % save_path)
    while True:
        prompt = "%s words: " % n_input
        sentence = 'When I Queen Mab within my fancy viewed, My'
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(64):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,rev_dict[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
            break
        except:
            print("Word not in dictionary")


# In[ ]:





# Sir Charles into my chamber coming in, When I was writing of my Fairy Queen; I praysaid hewhen Queen Mab you do see Present my service to her Majesty: And tell her I have heard Fame's loud report Both of her beauty and her stately court. When I Queen Mab within my fancy viewed, My thoughts bowed low, fearing I should be rude; Kissing her garment thin which fancy made, I knelt upon a thought, like one that prayed; And then, in whispers soft, I did present His humble service which in mirth was sent; Thus by imagination I have been In Fairy court and seen the Fairy Queen.

# In[ ]:





# In[ ]:




