#!/usr/bin/env python
# coding: utf-8

# I try to do classification with neural networks, but the lightGBM result is always better than NN result. 
# In the following example, for simplicity, I use only one feature and as you can see I can't reach lightGBM level of ROC. I have tried to use RNN cells, LSTM... batch normalization.. different data transformation - result is always the same, lightGBM result much more better.
# 
# Maybe someone uses neural networks and can share an information how we can improve the result. Should I do some specific transformation of the data, or maybe should I use some special network architecture.. or something else.. 
# any advice will be helpful. 
# Thank you.

# In[16]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[17]:


print('Importing data...')
data = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')

print('preparation..')
y = data['TARGET']
del data['TARGET']
cl = [
# 'AMT_INCOME_TOTAL',
 'AMT_CREDIT',]

data = data[cl]
test = test[cl]

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

#Impute missing
print('Imputing data...')
data = data.fillna(value = -1)
test = test.fillna(value = -1)

#Create new, balanced train set
data['y'] = y
data_all_ones = data[data.y==1]
data_all_zeros = data[data.y==0]
data_all_zeros2 = data_all_zeros.iloc[0:data_all_ones.shape[0],:]
data = pd.concat([data_all_ones,data_all_zeros2], axis = 0)
y= data.y
del data['y']


#Scale data to feed Neural Net
print('scaling...')
scaler = MinMaxScaler().fit(data)
data = scaler.transform(data)
test= scaler.transform(test)

print('finished')


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(data, y, test_size=0.15, shuffle=True, random_state=42)

X_train_a = X_train
y_train_a = y_train.values.reshape(-1,1)

X_valid_a = X_valid
y_valid_a = y_valid.values.reshape(-1,1)

X_test_a = test


# **Simple DNN with 2 hidden layers**

# In[24]:


#tensorflow model
import tensorflow as tf
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
reset_graph()

n_inputs = X_train.shape[1]
n_neurons = 10
n_outputs = 1

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.int32, shape=[None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing



#he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
he_init = tf.contrib.layers.xavier_initializer()

o_input = tf.layers.dense(X, n_inputs, activation=tf.nn.elu)
o_input = tf.layers.dropout(o_input, rate=0.5, training=tf_is_training)   # drop out 50% of inputs

o_hidden_1 = tf.layers.dense(o_input, n_neurons, activation=tf.nn.elu, kernel_initializer=he_init)
o_hidden_1 = tf.layers.dropout(o_hidden_1, rate=0.5, training=tf_is_training)   # drop out 50% of inputs

o_hidden_2 = tf.layers.dense(o_hidden_1, n_neurons, activation=tf.nn.elu, kernel_initializer=he_init)
o_hidden_2 = tf.layers.dropout(o_hidden_2, rate=0.5, training=tf_is_training)   # drop out 50% of input

logits = tf.layers.dense(o_hidden_2, n_outputs)
y_proba = tf.nn.sigmoid(logits)

y_as_float = tf.cast(y, tf.float32)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

y_pred_correct = tf.equal(y_pred, y)
accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))

init = tf.global_variables_initializer()

n_ecpochs = 100
batch_size = 1000

def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_ecpochs):
        for iteration in range(len(X_train_a) // batch_size):
            X_batch, y_batch = random_batch(X_train_a, y_train_a, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, tf_is_training: True})
            
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, tf_is_training: False})
        
        acc_val = accuracy.eval(feed_dict={X: X_valid_a, y: y_valid_a, tf_is_training: False})
        y_proba_val = y_proba.eval(feed_dict={X:X_valid_a, y: y_valid_a, tf_is_training: False})
        print(epoch, "Train accuracy:", acc_train, 'Validation accuracy:',acc_val)
    
   # acc_test = accuracy.eval(feed_dict={X: data_test_a})
    y_proba_test = y_proba.eval(feed_dict={X: X_test_a, tf_is_training: False})
    print(epoch, "Test prob:", y_proba_test)


# **Simple LightGBM Classifier**

# In[20]:


from lightgbm import LGBMClassifier

clf = LGBMClassifier(
        learning_rate=0.05,
    )
    
clf.fit(X_train_a, y_train_a.ravel(), 
            eval_set= [(X_train_a, y_train_a.ravel()), (X_valid_a, y_valid_a.ravel())], 
            eval_metric='auc', verbose=250, early_stopping_rounds=250
           )
    
y_pred_p = clf.predict_proba(X_valid_a, num_iteration=clf.best_iteration_)[:, 1]


# In[21]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import matplotlib.pyplot as plt


# Plot data
def generate_results(y_test, y_score):
    # print(y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


# **DNN ROC Curve**

# In[22]:


generate_results(y_valid, y_proba_val)


# **LightGBM ROC Curve**

# In[23]:


generate_results(y_valid.ravel(), y_pred_p)


# In[ ]:




