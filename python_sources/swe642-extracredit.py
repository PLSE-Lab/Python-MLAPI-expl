#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST Classification using Tensorflow

# # Importing Libraries:

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import OneHotEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu


# # Import & Pre-Process Data:

# In[29]:


train_csv = pd.read_csv(root_path+'/fashion-mnist_train.csv') 
test_csv = pd.read_csv(root_path+"/fashion-mnist_test.csv")
print("\nTraining Data:\n", train_csv.head())
print("\nTesting Data:\n", test_csv.head())


# ### Extract Training Labels:
# 
# We use One-Hot Encoding to extract Training Labels

# In[30]:


train_labels = np.array(train_csv['label'])
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
train_labels = train_labels.reshape(len(train_labels), 1)
TRAIN_LABELS_onehot_encoded = onehot_encoder.fit_transform(train_labels)
print("Training Labels Shape: ",len(TRAIN_LABELS_onehot_encoded),"x",len(TRAIN_LABELS_onehot_encoded[0]))


# ### Extract Training Data:
# 
# We extract training data into List of Lists:

# In[31]:


train_data = []
for i in range(len(train_csv)):
    imageArray = np.reshape(np.array(train_csv.iloc[i,1:]), (28,28))
    train_data.append(imageArray)
    percentage = 100*i/len(train_csv)

print("Training Data Shape:", len(train_data),"x",len(train_data[0]),"x",len(train_data[0][0]))


# ### Extract Testing Labels:
# We use One-Hot Encoding to extract Testing Labels

# In[32]:


test_labels = np.array(test_csv['label'])
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
test_labels = test_labels.reshape(len(test_labels), 1)
TEST_LABELS_onehot_encoded = onehot_encoder.fit_transform(test_labels)
print("Testing Labels Shape:", len(TEST_LABELS_onehot_encoded),"x",len(TEST_LABELS_onehot_encoded[0]))


# ### Extract Testing Data:
# 
# We extract testing data into a List of Lists

# In[33]:


test_data = []
for i in range(len(test_csv)):
    imageArray = np.reshape(np.array(test_csv.iloc[i,1:]), (28,28))
    test_data.append(imageArray)
    percentage = 100*i/len(test_csv)
        
print("Testing Data Shape:", len(test_data),"x",len(test_data[0]),"x",len(test_data[0][0]))


# ### Image Representation of Train and Test Data:

# In[34]:


plt.subplot(121)
plt.imshow(train_data[0], cmap='gray')
plt.title('train data')
plt.subplot(122)
plt.imshow(test_data[0],cmap='gray')
plt.title('test data')
plt.show()


# ### Storing in variables:
# 
# We store the training data in train_X,<br>
# We store the testing data in test_X,<br>
# We store the training labels in train_y,<br>
# We store the testing labels in test_y<br>

# In[35]:


train_X = np.reshape(train_data,(-1, 28, 28, 1))
test_X = np.reshape(test_data,(-1,28,28,1))
train_y = TRAIN_LABELS_onehot_encoded
test_y = TEST_LABELS_onehot_encoded

print("\nTraining Data Shape: ",train_X.shape,"\nTesting Data Shape: ",test_X.shape)
print("\nTraining Labels Shape: ", train_y.shape, "\nTesting Labels Shape: ", test_y.shape)


# # Creating CNN:

# ### Initializing Network Parameters:
# training_iters: Number of times we train our model<br>
# learning rate: The rate at which the model learns data<br>
# batch size: The number of imagespassed into the model for every batch<br>
# number of inputs: All images are 28 x 28 x 1<br>
# number of classes: There are 10 classes<br>

# In[54]:


training_iters = 60 
learning_rate = 0.001 
batch_size = 128
n_input = 28
n_classes = 10


# ### Defining Placeholders:
# 
# We define the placeholders for batch data and batch labels.<br>
# The placeholder to store the batch's data/images : 'x' has the dimensions 128 x 28 x 28 x 1<br>
# The placeholder to store the batch's labels : 'y' has the dimensions 128 x 10<br>
# 

# In[55]:


tf.reset_default_graph()
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])


# ### Defining our CNN and Maxpool Model:
# #### CNN:
# We pass out inputs 'x' and Filters 'W' and define our strides for each dimension into a CNN.<br>
# We then add the output with bias 'b' and send it to an activation function called RELU.
# #### Maxpool:
# We pass our output from CNN and a filter size into a Maxpool Layer. 

# In[56]:


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


# ### Defining our Weights and Biases:
# 
# The weights for a <b>CNN</b> are nothing but the filters.<br>The filter is a 4-D tensor defined as:<br>
# [input data rows, input data columns, input data channels, output channels]<br>
# The biases for a <b>CNN</b> is a 1-D tensor defined as:<br>
# [output channels]<br>
# <br><br>
# The weights for a <b>dense layer</b> are defined as:<br>
# [flattened output, output channels]
# 

# In[57]:


weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}


# ### Defining our Neural Network:
# 
# We pass our input data into a CNN and a Maxpool Layer 3 times.<br>
# The output of it is sent into a fully connected layer.
# 
# The Drop-Out layers are used to prevent overfit

# In[58]:


def conv_net(x, weights, biases):  

    # CNN 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling
    conv1 = maxpool2d(conv1, k=2)
    #Drop-Out
    conv1 = tf.nn.dropout(conv1, rate=0.25)

    # CNN 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling
    conv2 = maxpool2d(conv2, k=2)
    # Drop-Out
    conv2 = tf.nn.dropout(conv2, rate=0.25)

    # CNN 3
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling
    conv3 = maxpool2d(conv3, k=2)
    #Drop-Out
    conv3 = tf.nn.dropout(conv3, rate=0.25)

    # Fully connected layer
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# ### Defining out Optimizer:
# We will be using AdamOptimizer to adjust our weights and biases.

# In[59]:


pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# ### Prediction, Accuracy & Variable Initialize:
# Check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
# <br><br>
# Calculate accuracy across all the given images and average them out.
# <br><br>
# Initialize the variables

# In[60]:


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


# ### Training the model:

# In[65]:


predictions = []
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " +                       "{:.6f}".format(loss) + ", Training Accuracy= " +                       "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()
    predictions = sess.run(pred, feed_dict={x: test_X})


# ### Plotting Train vs Test Loss:

# In[72]:


plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()


# We can see that after applying drop-out our model is not overfit. The Test Loss is almost decreasing linearly with Train Loss****

# ### Train vs Test Accuracy:

# In[75]:


plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.legend()
plt.figure()
plt.show()


# We can see that after applying drop-out our model is not overfit. The Test Accuracy is almost increasing linearly with Train Accuracy.
