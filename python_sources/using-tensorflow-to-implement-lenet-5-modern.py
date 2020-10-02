#!/usr/bin/env python
# coding: utf-8

#   After finish Andrew Ng's Deeplearning course CNN chapter, I think is time to practice, so I write this kernel for sharing my practice and memorizing it.
#   
#   This kernel using Tensorflow to implement modern version of Le Net-5, After I finish this kernel, I think I start to familiar with style of Tensorflow, you first need to well define model achitechture and operations ,after that your need to use a "Session" to interact with you model, a "Session" kind like a console, a switch, a button, you use it to make your "DeepLearning Machine" start running.

# ### Load Data

# In[ ]:


# import modules
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


# inspecting first five rows
df_train.head()


# In[ ]:


print(df_train.shape)


# ### Check label distribution

# All labels seem fairly distributed.

# In[ ]:


sns.countplot(df_train.label)


# ### Normalizing data

# In[ ]:


# extract label from train dataset
labels = df_train[['label']]
df_train = df_train.drop('label',axis = 1)


# In[ ]:


# Normalization
df_train = df_train.astype('float32')/255
df_test = df_test.astype('float32')/255


# In[ ]:


train = df_train.values.reshape(-1,28,28,1)


# In[ ]:


# visualize three random pictures

plt.subplot(131)
plt.imshow(train[0][:,:,0],cmap=cm.binary)

plt.subplot(132)
plt.imshow(train[10][:,:,0],cmap=cm.binary)

plt.subplot(133)
plt.imshow(train[100][:,:,0],cmap=cm.binary)


# In[ ]:


labels['label'] = labels['label'].astype('str')
labels = pd.get_dummies(labels)


# ### Train test split

# In[ ]:


# setting random_state as 189 (the instance labels we visualize above :-> )
X_train,X_test,y_train,y_test = train_test_split(
    df_train,labels,
    test_size = 0.10, 
    random_state = 189,
    shuffle = True)


# In[ ]:


print('Train data size:')
print('X_train:',X_train.shape)
print('y_train:',y_train.shape)

print('Test data size:')
print('X_test:',X_test.shape)
print('y_test:',y_test.shape)


# In[ ]:


image_size = X_train.shape[1]
label_size = y_train.shape[1]


# In[ ]:


label_size


# In[ ]:


image_size


# In[ ]:


X_train.head()


# ### Constructing Le Net-5 (modern)

#   [Le Net-5](http://yann.lecun.com/exdb/lenet/index.html) is one kind architecture of Convolutional Neural Network (CNN) which announced by [Yann LeCun's paper](moz-extension://9bdee2ec-2bf8-4371-b383-f3c995e5149b/static/pdf/web/viewer.html?file=http%3A%2F%2Fyann.lecun.com%2Fexdb%2Fpublis%2Fpdf%2Flecun-01a.pdf) at 1998. It first came out for sloving Handwriting Recognition problem.This Kernel use its modern version.

# ![blockchain](data:image/png;base64,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)

# (sorry for my terrible drawing...)
# 
# The drawing below only first convolutional layer setting (padding = "SAME")  to make sure model look like Le Net-5 since its image INPUT shape is 32*32,  other layers all setting (padding = "VALID").
# 
# This Handdrawing picture shows CNN achitecture of this Kernel.
# 
# Compare to classic Le Net-5, I change activate function from sigmoid to ReLu, pooling layer filter shape add third dimension according to input shape.
# 

# ![Le Net-5 Modern Version](https://img-blog.csdnimg.cn/20190818171311738.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMTI1MjQ2,size_16,color_FFFFFF,t_70)

# In[ ]:


# Inputs and Outputs
x = tf.placeholder('float', shape=[None, image_size])
y = tf.placeholder('float', shape=[None, label_size])

# weights init

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# convolution definition

def conv2d(x, W, padding='VALID' ):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

# pooling definition
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



# In[ ]:


# settings
LEARNING_RATE = 1e-4
       
    
DROPOUT = 0.7
BATCH_SIZE =200


# ### Constrct Le Net-5 Convolution Neural Network

# In[ ]:


# shape = width*height*channels


# input     shape:28*28*1
image = tf.reshape(x, [-1,28 , 28,1])

# layer 1 convolution     shape:28*28*6
W_conv1 = weight_variable([5,5,1,6])
b_conv1 = bias_variable([6])

h_conv1 = tf.nn.relu(conv2d(image, W_conv1,padding = 'SAME') + b_conv1)

# layer 2 average pooling     shape:14*14*6
h_pool1 = max_pool_2x2(h_conv1)

# layer 3 convolution     shape:10*10*16
W_conv2 = weight_variable([5,5,6,16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# layer 4 average pooling     shape:5*5*16
h_pool2 = max_pool_2x2(h_conv2)

# layer 5 fully connect with 120 neurons     120 dim array
W_fc1 = weight_variable([5*5*16, 120])
b_fc1 = bias_variable([120])

h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*16])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

## add dropout for regularization
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# layer 6 fully connect with 84 neurons     84 dim array
W_fc2 = weight_variable([120, 84])
b_fc2 = bias_variable([84])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

## add dropout for regularization

h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)


# layer 7 softmax output     10 dim array
W_readout = weight_variable([84,label_size])
b_readout = bias_variable([label_size])

y_prob = tf.nn.softmax(tf.matmul(h_fc2_drop, W_readout) + b_readout)


# ### Optimisation  and Evaluation

# In[ ]:


# Cost function: Cross Entropy Loss
cross_entropy = -tf.reduce_sum(y*tf.log(y_prob))

# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_prob,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


# In[ ]:


predict = tf.argmax(y_prob,1)


# ### Train Approach

# In[ ]:


epochs_completed = 0
index_in_epoch = 0
num_examples = X_train.shape[0]

train = X_train.values
label = y_train.values

epoches = 20
# serve data by batches
def next_batch(batch_size):
    
    global train
    global label
    global index_in_epoch
    global epochs_completed
    

    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train = train[perm]
        label = label[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train[start:end], label[start:end]


# In[ ]:


# extract values from DataFrame as numpy array
validation_features = X_test.values
validation_labels = y_test.values
test_features = df_test.values

# setting epoch
EPOCHES = 50

# initialize all variables 
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# start trainning
# per epoch train
for epoch in range(EPOCHES):
    # per mini-batch train
    for mini_batch in range(num_examples // BATCH_SIZE):
        # get mini-batch
        batch_xs, batch_ys = next_batch(BATCH_SIZE)
        
        # forward propagation and backward propagation, optimize with Adam gradient descent
        
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob:DROPOUT})
    
    # caculate train dataset accuracy and validation dataset accuracy
    train_accuracy = sess.run(accuracy, feed_dict={x: train, y: label, keep_prob:1.0})
    validation_accuracy = sess.run(accuracy, feed_dict={x: validation_features, y: validation_labels, keep_prob:1.0})
    
    print("epoch "+ str(epoch)+ ': '+'train_accuracy: '+ str(train_accuracy),
             'validation_accuracy: '+ str(validation_accuracy))


# ### Prediction

# In[ ]:


predicted_lables  = sess.run(predict,feed_dict={x:test_features,keep_prob: 1.0})


# In[ ]:


predicted_lables


# In[ ]:


id = np.array(list(df_test.index)) +1


# In[ ]:


submission = pd.DataFrame({'ImageId':list(id),'Label':list(predicted_lables)})


# In[ ]:


submission.to_csv('submission.csv',index = False)


# ###  Never forget to close your session

# In[ ]:


sess.close()

