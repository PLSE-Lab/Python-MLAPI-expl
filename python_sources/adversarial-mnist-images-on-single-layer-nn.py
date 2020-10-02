#!/usr/bin/env python
# coding: utf-8

# This notebook attempts to make adversarial examples on MNIST single layer model by implementing FGSM method. Please help in pointing out any issue.     
# Motivation came from kernel   https://www.kaggle.com/allunia/example-attacking-logistic-regression

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print("Avaiable Datasets: ")
print(check_output(["ls", "../input"]).decode("utf8"))
print("Files in Digit Recognizer Dataset: ")
print(check_output(["ls", "../input/digit-recognizer"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


# read the data
df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")
df_train.head()


# In[ ]:


# Preprocessing..not doing train-test split
Y = df_train['label']
X = df_train.drop('label', axis=1)
# normalize input image between -1 and 1
X = X.apply(lambda x: x*2.0/255.0 - 1.0)


# In[ ]:


# explore the digits
fig1, ax1 = plt.subplots(1,15, figsize=(15,10))
for i in range(15):
    ax1[i].imshow(X.iloc[i].values.reshape((28,28)), cmap='gray')
    ax1[i].axis('off')
    ax1[i].set_title(Y[i])


# In[ ]:


# create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
# Define actual outputs
y_ = tf.placeholder(tf.float32, [None, 10])


# In[ ]:


# define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[ ]:


# Batch Generator
def get_batch(BATCH_SIZE=64):
    while (1):
        for i in range(0, len(X), BATCH_SIZE):
            x_train, y_train = X[i:i+BATCH_SIZE], Y[i:i+BATCH_SIZE]
            yield (x_train, y_train)


# In[ ]:


# Training
gen = get_batch(64)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(500):
    batch_xs, batch_ys = next(gen)
    batch_ys = np.eye(10)[batch_ys]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if i%50 == 0:
        print('ACCURACY:', sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


# In[ ]:


# Apply FGSM
grad = tf.gradients(cross_entropy, x)
signed_grad = tf.sign(grad)
scaled_signed_grad = 0.8 * signed_grad
# generate adverserial example and rescale to get image between 0 and 255
adv_x = (((tf.clip_by_value((x + scaled_signed_grad), -1, 1)+ 1.0) * 0.5) * 255.0)


# In[ ]:


# Let us generate adversarial of 7th image in data i.e. 
fig, ax = plt.subplots(1,1, figsize=(1,10))
ax.imshow(X.iloc[6].values.reshape((28,28)), cmap='gray')
ax.axis('off')
ax.set_title("True")


# In[ ]:


# generate adversarial
one_hot_output = np.eye(10)[Y[6]].reshape(1,10)
advsarial_out = sess.run(adv_x, feed_dict={x: X.iloc[6].values.reshape(1,784), y_:one_hot_output})


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(1,10))
ax.imshow(advsarial_out.reshape((28,28)).reshape((28,28)), cmap='gray')
ax.axis('off')
ax.set_title("Fake")


# Something is amiss here. I was trying to add non-targeted perturbation but 7 is changing to 8.     
# What am I missing?

# In[ ]:


# Original Prediction
original_prediction = sess.run(y, feed_dict={x: X.iloc[6].values.reshape(1,784), y_:one_hot_output})
print('Original Prediction',np.argmax(original_prediction, 1))


# In[ ]:


# Adversarial Prediction
adversarial_prediction = sess.run(y, feed_dict={x: advsarial_out.reshape(1,784), y_:one_hot_output})
print('Fake Prediction', np.argmax(adversarial_prediction, 1))

