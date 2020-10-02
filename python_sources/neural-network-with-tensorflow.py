#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# ## Load Data

# In[3]:


Image_path='../input/Sign-language-digits-dataset/X.npy'
X = np.load(Image_path)
label_path='../input/Sign-language-digits-dataset/Y.npy'
Y = np.load(label_path)

print("X Dataset:",X.shape)
print("Y Dataset:",Y.shape)


# The X dataset has 2062 images with 64x64 pixel each.
# <br>
# The Y dataset has 2062 labels with 10 different classes (0-9).

# ## Split Dataset
# The test_size represents the proportion of the dataset that belongs to the test split. 
# <br>The random_state makes shure, that the train-test-split is the same, even when you restart the notebook.

# In[4]:


from sklearn.model_selection import train_test_split
test_size = 0.15
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=42)

img_size = 64
channel_size = 1
print("Training Size:", X_train.shape)
print(X_train.shape[0],"samples - ", X_train.shape[1],"x",X_train.shape[2],"grayscale image")

print("\n")

print("Test Size:",X_test.shape)
print(X_test.shape[0],"samples - ", X_test.shape[1],"x",X_test.shape[2],"grayscale image")


# The splitted training size has 1752 grayscale images with 64x64 pixels each.
# <br>
# The splitted test size has 310 grayscale images with 64x64 pixels each.

# ## Sanity Check - Test Images

# In[5]:


print('Test Images:')
n = 10
plt.figure(figsize=(20,20))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_train[i].reshape(img_size, img_size))
    plt.gray()
    plt.axis('off')


# ## Array For Training
# 

# In[6]:


x_result_array = np.zeros(shape=(1024,0))

width = 32
height = 32

for i in range (0,len(X_train)):
    img = X_train[i]
    img1 = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
    t = img1.reshape(1,1024).T
    x_result_array = np.concatenate([x_result_array,t], axis=1)
    
x_result_array = x_result_array.T
print(x_result_array.shape)
print(Y_train.shape)


# ## Array For Testing

# In[7]:


x_test_array = np.zeros(shape=(1024,0))

width_test = 32
height_test = 32

for i in range (0,len(X_test)):
    img = X_test[i]
    img1 = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
    t = img1.reshape(1,1024).T
    x_test_array = np.concatenate([x_test_array,t], axis=1)

x_test_array = x_test_array.T
print(x_test_array.shape)
print(Y_test.shape)


# In[10]:


import tensorflow as tf


# ## Placeholder

# In[11]:


X = tf.placeholder(tf.float32,name="X-Input")
Y = tf.placeholder(tf.float32,name="Y-Output")


# ## Hyperparameter

# In[12]:


learning_rate = 0.1
epochs = 15001
x_inp = x_result_array.shape[1] #Neurons in input layer -> Dimensions of feature input
n_1 = 1000 #Neurons in first hidden layer
y_out = 10 #Neurons in output layer -> 10 Classes


# ## Initialize The Weights and Biases

# In[13]:


w1 = tf.Variable(tf.random_normal([x_inp,n_1]),name="w1") #Input -> First Hidden Layer
 
w2 = tf.Variable(tf.random_normal([n_1,y_out]),name="w2") #First Hidden Layer -> Output Layer

b1 = tf.Variable(tf.zeros([n_1]),name="b1") #Input -> First Hidden Layer
b2 = tf.Variable(tf.zeros([y_out]),name="b2") #First Hidden Layer -> Output Layer


# ## Neural Network Model

# In[15]:


#First Hidden Layer
A1 = tf.matmul(X,w1)+b1
H1 = tf.nn.relu(A1)

#Output Layer
logit = tf.add(tf.matmul(H1,w2),b2)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=Y)

#Cost with L2-Regularizer
cost = (tf.reduce_mean(cross_entropy))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Prediction
y_pred = tf.nn.softmax(logit)

pred = tf.argmax(y_pred, axis=1 )


# ## Start Tensorflow Session
# 

# In[16]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# ## Sanity Check

# In[17]:


print("Initialized w1:")
print(sess.run(w1).shape)
print("\n")

print("Initialized w2:")
print(sess.run(w2).shape)
print("\n")

print("Initialized b1:")
print(sess.run(b1).shape)
print("\n")

print("Initialized b2:")
print(sess.run(b2).shape)
print("\n")

print("Cost1:")
print(sess.run([cost],feed_dict={X:x_result_array,Y:Y_train}))


# ## Training

# In[18]:


p = []
for epoch in range (0,epochs):
    values = sess.run([optimizer,cost,cross_entropy,w1,w2,b1,b2], feed_dict={X:x_result_array,Y:Y_train})

    if epoch%1000 ==0:
        print("Epoch:", epoch)
        print("Cost:",values[1])
        print("_____")
        p.append(values[1])


# In[21]:


plt.plot(p[1:])
plt.title("Cost Decrease")


# ## Next Step
# * Add Test Pipeline
# * Accuracy
# * Improvements

# In[ ]:




