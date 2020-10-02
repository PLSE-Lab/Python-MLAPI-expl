#!/usr/bin/env python
# coding: utf-8

# TensorFow is Fun !!!
# This is a beginner's work in both TensorFlow and Deep learning. I have used feed forward neural networks with one hidden layer.

# In this kernel I have used Google's amazing library TensorFlow for training a simple feed forward neural network on handwritten digits dataset.

# # We will start by importing the necessary libraries

# In[ ]:



import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


# Let's import the test dataset
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


# Import the training dataset by making a function read_dataset.
def read_dataset():
    df = pd.read_csv("../input/train.csv")
    
    # In X we will have all the values of the pixels.
    # Y will have the label that is what number is it.
    X = df[df.columns[1:785]].values # features
    y = df[df.columns[0]] # labels
   

    # We need to encode our dataset using simple encoding.
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    return(X,Y)


# In[ ]:


# Using one hot encoding.
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# In[ ]:


X, Y = read_dataset()

# Shuffle the dataset to mix up the rows.
X, Y = shuffle(X, Y, random_state=1)

# Spliiting the dataset into train and test with 20% being the test size and 80% is the training size.
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# The rate by which our dataset will learn.
learning_rate = 0.01

# Epochs is basically the number of iterations.
training_epochs = 20
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 10


# In[ ]:


# I have used  one hidden layer with the number of neurons equal to 650.
n_hidden_1 = 650


# In[ ]:


batch_size = 128


# # Applying TensorFlow

# **x and y_ are placeholders whereas weights and biases will be variables.**

# In[ ]:


x = tf.placeholder(tf.float32,[None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class])) # Weights
b = tf.Variable(tf.zeros([n_class])) # Biases
y_ = tf.placeholder(tf.float32,[None,n_class])


# **Using Sigmoid Activation for hidden layer and linear activation for output layer**

# In[ ]:


def multilayer_perceptron(x, weights, biases):

    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

   # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


# **Making dictionary for the weights and biases**

# In[ ]:


weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_class]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}


# In[ ]:


# To initialize the variables
init = tf.global_variables_initializer()
y = multilayer_perceptron(x, weights, biases)


# ** Using gradient descent optimizer**

# In[ ]:


cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)


# We are running the model for 20 epochs. Epoch basically means an iteration.

# In[ ]:


# Calculate the cost and the accuracy for each epoch

# List for mean square error.
mse_history = []

# List for accuracy.
accuracy_history = []

for epoch in range(training_epochs):
    
    
    total_batch = int(train.shape[0]/batch_size)
    for i in range(total_batch):
            avg_cost = 0
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            sess.run(training_step, feed_dict={x: train_x, y_: train_y})
            cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
            
            avg_cost += cost/total_batch
            cost_history = np.append(cost_history, cost)
            
    
    # Calculating the number of correct predictions.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Calculating the accuracy.
    print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
    pred_y = sess.run(y, feed_dict={x: test_x})
    
    # Calculating the mean square error.
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)

    print('epoch : ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

#save_path = saver.save(sess, model_path)
#print("Model saved in file: %s" % save_path)

# Plot mse and accuracy graph

plt.plot(mse_history, 'r')
plt.show()
plot.title("Mean Square Error")
plt.plot(accuracy_history)
plot.title("Accuracy")
plt.show()


# In[ ]:


# Print the final accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))

# Print the final mean square error

pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))


# Applying on the validation set.

# Any suggestions are welcomed.  If there are any mistakes please do inform. If you like the kernel please do cast an upvote.
# 
# 
# *Ayushi Asthana
# March 2018*

# In[ ]:




