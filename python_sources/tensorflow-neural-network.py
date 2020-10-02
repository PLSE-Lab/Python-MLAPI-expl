#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.describe()


# In[ ]:


train.head()


# percent_atom_al + percent_atom_ga + percent_atom_in == 1

# In[ ]:


np.all(np.abs(train.loc[:, 'percent_atom_al'] + train.loc[:, 'percent_atom_ga'] + train.loc[:, 'percent_atom_in'] - 1) <= 0.001)


# In[ ]:


train = train.drop(['spacegroup', 'percent_atom_in'], axis=1)
test = test.drop(['spacegroup', 'percent_atom_in'], axis=1)


# **Data Preparation**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


t1 = 'formation_energy_ev_natom'
t2 = 'bandgap_energy_ev'

X_train, X_validation = train_test_split(train, test_size=0.3)

y1_train = np.log1p(X_train[t1][:,np.newaxis])
y2_train = np.log1p(X_train[t2][:,np.newaxis])
X_train = X_train.drop(['id', t1, t2], axis=1)

y1_validation = np.log1p(X_validation[t1][:, np.newaxis])
y2_validation = np.log1p(X_validation[t2][:, np.newaxis])
X_validation = X_validation.drop(['id', t1, t2], axis=1)

print(X_train.shape, y1_train.shape, y2_train.shape)
print(X_validation.shape, y1_validation.shape, y2_validation.shape)


# **Visualization**

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.subplot(1, 2, 1)
plt.scatter(range(len(y1_train)), y1_train)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y2_train)), y2_train)

plt.show()


# **TensorFlow**

# In[ ]:


import tensorflow as tf


# **Neural Network**

# In[ ]:


tf.set_random_seed(1)
np.random.seed(1)

tf.reset_default_graph() 

n_units = 16
n_layers = 8

activation = tf.tanh
#activation = tf.nn.relu
#activation = tf.nn.sigmoid


# In[ ]:


tf_is_training = tf.placeholder(tf.bool, None)

tf_x = tf.placeholder(tf.float32, (None, X_train.shape[1]), name='tf_x')
tf_y1 = tf.placeholder(tf.float32, (None, 1), name='tf_y1')
tf_y2 = tf.placeholder(tf.float32, (None, 1), name='tf_y2')


# In[ ]:


tf_xn = tf.layers.batch_normalization(tf_x, training=tf_is_training, name='tf_xn')


# In[ ]:


def add_norm_layer(inputs, nunits, activation, training=False, name=None):
    l = tf.layers.dense(tf_xn, n_units, activation=activation, name=name)
    ln = tf.layers.batch_normalization(l, training=training, name=name + 'n')
    
    return ln


# In[ ]:


# Formation energy
l = tf_xn
for i in range(n_layers):
    l = add_norm_layer(l, n_units, activation, training=tf_is_training, name='l%s_y1' % (i + 1))
output_y1 = tf.layers.dense(l, 1, name='output_y1')

# Bandgap energy
l = tf_xn
for i in range(n_layers):
    l = add_norm_layer(l, n_units, activation, training=tf_is_training, name='l%s_y2' % (i + 1))
output_y2 = tf.layers.dense(l, 1, name='output_y2')


# In[ ]:


# loss
loss_y1 = tf.sqrt(tf.losses.mean_squared_error(tf_y1, output_y1))
loss_y2 = tf.sqrt(tf.losses.mean_squared_error(tf_y2, output_y2))
loss = (loss_y1 + loss_y2) / 2.0

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=1.0)
optimizer = tf.train.AdamOptimizer(learning_rate=0.004)
train_op = optimizer.minimize(loss)


# **Training**

# In[ ]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:


loss_data = []

for step in range(500):
    # training loss
    _, lt = sess.run([train_op, loss], {tf_x: X_train, tf_y1: y1_train, tf_y2: y2_train, tf_is_training: True})
    
    # validation loss
    lv = sess.run(loss, {tf_x: X_validation, tf_y1: y1_validation, tf_y2: y2_validation, tf_is_training: True})
    
    loss_data.append([step, lt, lv])

loss_data = np.array(loss_data)

print(loss_data[-1][1:])

plt.plot(loss_data[:, 0], loss_data[:, 1], 'r-')
plt.plot(loss_data[:, 0], loss_data[:, 2], 'b-')
plt.show()


# **Validation**

# In[ ]:


loss, pred_y1, pred_y2 = sess.run([loss, output_y1, output_y2], {tf_x: X_validation, tf_y1: y1_validation, tf_y2: y2_validation, tf_is_training: True})

print('loss:', loss)

# Formation energy
m_y1 = max(y1_validation.max(), pred_y1.max())

ax1_y1 = plt.subplot(2, 2, 1)
ax1_y1.set_ylim([0, m_y1])
plt.scatter(range(len(y1_validation)), y1_validation)

ax2_y1 = plt.subplot(2, 2, 2)
ax2_y1.set_ylim([0, m_y1])
plt.scatter(range(len(pred_y1)), pred_y1, c='red')


# Bandgap energy
m_y2 = max(y2_validation.max(), pred_y2.max())

ax1_y2 = plt.subplot(2, 2, 3)
ax1_y2.set_ylim([0, m_y2])
plt.scatter(range(len(y2_validation)), y2_validation)

ax2_y2 = plt.subplot(2, 2, 4)
ax2_y2.set_ylim([0, m_y2])
plt.scatter(range(len(pred_y2)), pred_y2, c='red')

plt.show()


# **Submission**

# In[ ]:


# sample submission
sample = pd.read_csv('../input/sample_submission.csv')
sample.head()


# In[ ]:


X_test = test.drop(['id'], axis=1)

pred_y1, pred_y2 = sess.run([output_y1, output_y2], {tf_x: X_test, tf_is_training: True})

subm = pd.DataFrame()
subm['id'] = sample['id']
subm['formation_energy_ev_natom'] = np.expm1(pred_y1)
subm['bandgap_energy_ev'] = np.expm1(pred_y2)
subm.to_csv("subm.csv", index=False)

