#!/usr/bin/env python
# coding: utf-8

# Perform the necessary imports

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import re
import os
import cv2


# In[ ]:


data = "../input/train/"
train_dev_set = [data+idx for idx in os.listdir(path=data)]

def atoi(text):
    if text.isdigit() == True: #text contains digits only
        return int(text)
    else:
        return text
    
def natural_keys(text):
    return [atoi(idx) for idx in re.split(pattern='(\d+)', string=text)]

train_dev_set.sort(key=natural_keys)
train_dev_set = train_dev_set[0:1500] + train_dev_set[12500:14000]


# In[ ]:


def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = np.zeros(shape=(len(list_of_images), 150, 150, 3))
    y = np.zeros(shape=(1, len(list_of_images)))
    
    for (index, image) in enumerate(list_of_images):
        #read_image = np.array(ndimage.imread(image, flatten=False)) #deprecated in SciPy 1.0.0
        #my_image = scipy.misc.imresize(read_image, size=(150,150)) #deprecated in Scipy 1.0.0, will be removed in SciPy 1.2.0
        read_image = np.array(cv2.imread(image))
        my_image = cv2.resize(src=read_image, dsize=(150, 150))
        x[index] = my_image
    
    for (index, image) in enumerate(list_of_images):
        if 'dog' in image:
            y[:, index] = 1
        elif 'cat' in image:
            y[:, index] = 0
            
    return x, y


# In[ ]:


x_train_dev, y_train_dev = prepare_data(train_dev_set)


# In[ ]:


def numpy_array_properties(array):
    print("type:{}, shape:{}, dimensions:{}, size:{}, datatype:{}".format(type(array), array.shape, array.ndim, array.size, array.dtype))


# In[ ]:


numpy_array_properties(x_train_dev)
numpy_array_properties(y_train_dev)


# In[ ]:


fig = plt.figure()
img1 = fig.add_subplot(1,2,1) #1 row, 2 coulmns, image fills the 1st column
img1.imshow(x_train_dev[50]) 
img2 = fig.add_subplot(1,2,2) #1 row, 2 columns, image fills the 2nd column
img2.imshow(np.array(cv2.imread(train_dev_set[50])))


# In[ ]:


fig = plt.figure()
img1 = fig.add_subplot(1,2,1) #1 row, 2 coulmns, image fills the 1st column
img1.imshow(x_train_dev[2305]) 
img2 = fig.add_subplot(1,2,2)
img2.imshow(np.array(cv2.imread(train_dev_set[2305]))) #1 row, 2 columns, image fills the 2nd column


# In[ ]:


x_train_dev_flatten = x_train_dev.reshape(x_train_dev.shape[0], -1).T
x_train_dev_flatten_normalized = x_train_dev_flatten/255.
numpy_array_properties(x_train_dev_flatten_normalized)


# In[ ]:


#np.random.seed(2) #global seeding doesn't help cost convergence
permutation = list(np.random.permutation(x_train_dev_flatten_normalized.shape[1]))
x_train_dev_flatten_normalized_shuffled = x_train_dev_flatten_normalized[:, permutation]
y_train_dev_shuffled = y_train_dev[:, permutation]


# In[ ]:


numpy_array_properties(x_train_dev_flatten_normalized_shuffled)
numpy_array_properties(y_train_dev_shuffled)


# In[ ]:


x_train = x_train_dev_flatten_normalized_shuffled[:, 0:2600]
x_dev = x_train_dev_flatten_normalized_shuffled[:, 2600:3000]
y_train = y_train_dev_shuffled[:, 0:2600]
y_dev = y_train_dev_shuffled[:, 2600:3000]


# In[ ]:


numpy_array_properties(x_train)
numpy_array_properties(y_train)
numpy_array_properties(x_dev)
numpy_array_properties(y_dev)


# In[ ]:


print(x_train)
print("*"*100)
print(y_train)
print("*"*100)
print(x_dev)
print("*"*100)
print(y_dev)


# In[ ]:


#Before proceeding, plot the images and verify the labels(to validate shuffling has happened in synch between x and y, also for fun ;))
def recreate_image_from_numpy_array(x, y, idx):
    x = x*255. #undo normalize
    x = x.T.reshape(x.shape[1], 150, 150, 3) #undo flatten ex: x_dev.shape = (400, 150, 150, 3)
    print("Label for the below image is {}".format(int(np.squeeze(y[:, idx]))))
    plt.imshow(x[idx])


# In[ ]:


recreate_image_from_numpy_array(x_dev, y_dev, 397)


# In[ ]:


recreate_image_from_numpy_array(x_dev, y_dev, 394)


# **Data preprocessing and analysis ends. Deep learning begins!**

# In[ ]:


def create_placeholders(n_x, n_y, m):
    #X = tf.placeholder(name="X", shape=(n_x, None), dtype=tf.float32)
    #Y = tf.placeholder(name="Y", shape=(n_y, None), dtype=tf.float32)
    X = tf.placeholder(name="X", shape=(n_x, m), dtype=tf.float32)
    Y = tf.placeholder(name="Y", shape=(n_y, m), dtype=tf.float32)
    
    return X, Y


# In[ ]:


def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable(name="W1", shape=(125, 67500), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable(name="b1", shape=(125, 1), 
                         initializer=tf.zeros_initializer())
    W2 = tf.get_variable(name="W2", shape=(50, 125), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable(name="b2", shape=(50, 1), 
                         initializer=tf.zeros_initializer())
    W3 = tf.get_variable(name="W3", shape=(50, 50), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable(name="b3", shape=(50, 1), 
                         initializer=tf.zeros_initializer())
    W4 = tf.get_variable(name="W4", shape=(1, 50), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable(name="b4", shape=(1, 1), 
                         initializer=tf.zeros_initializer())
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2, "W3":W3, "b3":b3, "W4":W4, "b4":b4}
    return parameters


# In[ ]:


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    return Z4


# In[ ]:


def compute_cost(Z4, Y):
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(name="cost", logits=logits, labels=labels))
    return cost


# In[ ]:


def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.01, 
          minibatch_size = 16, num_epochs = 200, print_cost = True):
    
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    X, Y = create_placeholders(n_x=n_x, n_y=n_y, m=m)
    parameters = initialize_parameters()
    Z4 = forward_propagation(parameters=parameters, X=X)
    cost = compute_cost(Y=Y, Z4=Z4)
    
    #optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.99, epsilon=10**-8, 
                                       #learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        epoch_cost = 0
        for epoch in range(num_epochs):
            _ , epoch_cost = sess.run(fetches=[optimizer, cost], feed_dict={X:X_train, Y:Y_train})
            if print_cost == True and epoch%10 == 0:
                print("Cost after epoch {} is : {}".format(epoch, epoch_cost))
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel("Cost")
        plt.xlabel("Iterations(per tens)")
        plt.title("Learning rate: {}".format(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        correct_prediction = tf.equal(x=tf.round(x=tf.sigmoid(Z4)), y=Y_train)
        accuracy = tf.reduce_mean(tf.cast(x=correct_prediction, dtype="float"))
        
        print("Train accuracy:", accuracy.eval({X:X_train, Y:Y_train}))
        #print("Dev set accuracy:", accuracy.eval({X:X_dev, Y:Y_dev}))
        
        return parameters


# In[ ]:


parameters = model(X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




