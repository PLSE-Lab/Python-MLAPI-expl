#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network - Pneumonia Prediction Using X-Ray Images
# 
# Welcome to my third machine learning project! 
# 
# We will be using TensorFlow to create a convolutional neural network with two convolutional layers and one fully connected layer to predict pneumonia, given a chest X-ray image as the input. 
# 
# Let's start by importing the packages we will be using throughout the kernel:

# In[ ]:


import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import random
from random import randint
from pathlib import Path
import os
print(os.listdir("../input"))


# # Step 1 - Visualize our image data and split it into training and test sets
# 
# 
# The dataset has been split into 'train' and 'test'' folders.  Additionally, the images within these folders have been split into 'NORMAL' and 'PNEUMONIA' subfolders. 
# 
# First, we will go through the folders of the dataset and append each image into an array. Both the training and test set images will be placed into the same matrices for now (later to be randomly split into our training and test sets).  Note that each image will be resized to 64x64 to reduce computation time and to maintain consistency throughout our dataset.
# 
# Because this is a binary classification problem, each image will have an associated classification vector of the form (1,0), meaning the patient has pneumonia, or (0,1), meaning the patient does not have pneumonia. 
# 
# After our images have been pulled and classified, they will be normalized.  This is done to ensure that our gradients do not diverge during back propagation. 
# 
# Finally, our data will be split into training and test sets.  The training set will use 2/3 of the available X-ray images.

# In[ ]:


train_dir = '../input/chest_xray/chest_xray/train'
test_dir =  '../input/chest_xray/chest_xray/test'
X = []
Y = []

#Loop through the training and test folders, as well as the 'NORMAL' and 'PNEUMONIA' subfolders and append all images into array X.  Append the classification (0 or 1) into array Y.

for fileName in os.listdir(train_dir + "/NORMAL"): 
        img = cv2.imread(train_dir + "/NORMAL/" + fileName)
        if img is not None:
            Y.append(0)
            img = cv2.resize(img,(64,64))
            X.append(img)
    
for fileName in os.listdir(train_dir + "/PNEUMONIA"): 
        img = cv2.imread(train_dir + "/PNEUMONIA/" + fileName)
        if img is not None:
            Y.append(1)
            img = cv2.resize(img,(64,64))
            X.append(img)
            
for fileName in os.listdir(test_dir + "/NORMAL"): 
        img = cv2.imread(test_dir + "/NORMAL/" + fileName)
        if img is not None:
            Y.append(0)
            img = cv2.resize(img,(64,64))
            X.append(img)
    
for fileName in os.listdir(test_dir + "/PNEUMONIA"): 
        img = cv2.imread(test_dir + "/PNEUMONIA/" + fileName)
        if img is not None:
            Y.append(1)
            img = cv2.resize(img,(64,64))
            X.append(img)


# # Step 1.1 - Data visualization
# 
# Let's visualize our data before we split it into our training and test sets in the following steps:
# 
# * Observe an example of a positive and negative pneumonia diagnosed X-ray to see what our algorithm is trying to detect
# 
# * Count the number of pneumonia positive and negative results in our data set (i.e. our data's distribution)
# 
# * Create a bar graph of the positive and negative count as a simple visualization of our distribution
# 
# Let's start with our visualization:

# In[ ]:


print("This is an example of a patient X-ray who does not have pneumonia:")
normal = cv2.imread(test_dir + "/NORMAL/IM-0003-0001.jpeg")
plt.axis('off')
plt.imshow(normal)


# In[ ]:


print("This is an example of an X-ray of a patient diagnosed with pneumonia:")
pnumonia = cv2.imread(test_dir + "/PNEUMONIA/person15_virus_46.jpeg")
plt.axis('off')
plt.imshow(pnumonia)


# We have now seen an example of both positive and negative pneumonia X-rays.  Note the 'fog' like discoloration in the pneumonia positive X-ray.  This is essentially what we would like our algorithm to detect.
# 
# Let's see how our classifications are distributed throughout out dataset:

# In[ ]:


#Data visualization

pos = 0
neg = 0

for i in range(1,len(Y)):
    if Y[i] == 1:
        pos = pos + 1
    else:
        neg = neg + 1

objects = ('Positive', 'Negative')
y_pos = np.arange(len(objects))
performance = [pos,neg]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('X-Ray Diagnosis')
 
plt.show()

print("There are " +str(pos) +" pneumonia positive X-ray's in our data")
print("There are " +str(neg) +" pneumonia negative X-ray's in our data")


# # Step 1.2 - Normalizing and splitting our data
# 
# We can see our data is biased to pneumonia positive results.
# 
# It will be important to check the distribution of our training and test sets after we randomly shuffle and create them.  
# 
# In addition, to ensure our gradients do not diverge, we will normalize our image data.

# In[ ]:


#Normalize our images to ensure gradients do not diverge
X = np.array(X)/255

#Normalize our data by setting the mean to 0 and variance to 1.
X = (X - np.average(X,0))/np.std(X,0)

#Convert our Y vector into a categorical (e.g. 0 -> (0,1), 1 -> (1,0))           
from keras.utils.np_utils import to_categorical
Y = to_categorical(Y, num_classes = 2) #Randomly create our training and test sets using 2/3 of the data for training, and the remaining 1/3 for testing


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

pos_train =0
neg_train = 0
pos_test = 0
neg_test = 0


for i in range(1,len(Y_train)):
    if Y_train[i][0] == 1:
        pos_train = pos_train + 1
    else:
        neg_train = neg_train + 1
        
for i in range(1,len(Y_test)):
    if Y_test[i][0] == 1:
        pos_test = pos_test + 1
    else:
        neg_test = neg_test + 1
        
print("Positive in training: " + str(pos_train))
print("Negative in training: " + str(neg_train))
print("Train pos/neg ratio: " + str (pos_train/neg_train))
      
print("Positive in test: " + str(pos_test))
print("Negative in test: " + str(neg_test))
print("Test pos/neg ratio: " + str (pos_test/neg_test))


# Looks like our data has a similar distribution for both our training and test sets.

# # Step 2 - Set up TensorFlow variables
# 
# We will now set up our TensorFlow variables to be used in our forward propogation and cost function.
# 
# First, we set up placeholder values for our image and classification matrices.  
# 
# * X will represent our image matrix, which will have dimensions #samples x height x width x # filters
# 
# * Y will represent our classification matrix, which will have dimensions # samples x # classifications.  In our case, the # classifications is 2 (0 or 1)

# In[ ]:


def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])
    
    return X, Y


# Next, we will set up our randomly initialized weights for each of our layers.  Recall we are creating a convolutional neural network with two convolutional layers and one fully connected layer.  
# 
# Our weights must be initialized appropriately to correspond with our desired network.  The dimension of our weights, as initialized below, is (height, width, layers, # filters).  The following will be the structure of our convolutional neural network, and our weights will be initialized accordingly:
# 
# * The first convolutional layer will consisnt of 10 3-layered 3x3 filters.  A ReLu activation function will be applied after convolution.  Our weights for this layer should be of dimension (3,3,3,10)
# 
# * The second convolutional layer will consist of 8 10-layered 5x5 filters. A ReLu activation function will be applied after convolution. Our weights for this layer should be of dimension (5,5,10,8)
# 
# * Our final layer, the fully connected one, will flatten out the values from the second convolutional layer and apply a sigmoid activation
# 
# For simplicity, we will not use any pooling layers, nor will we use padding or strides > 1.
# 
# See below for a visual representation of our convolutional neural network for a single image:
# 
# <img src="https://imgur.com/ZDKAac5.png" width="1000px"/>

# In[ ]:



def initialize_parameters():
    W1 = tf.get_variable("W1",[3,3,3,10],initializer = tf.contrib.layers.xavier_initializer()) #Define our weight for the first convolutional layer
    W2 = tf.get_variable("W2",[5,5,10,8],initializer = tf.contrib.layers.xavier_initializer()) #Define our weight for the second convolutional layer
    
    parameters ={'W1':W1,
                'W2':W2,
                }
    
    return parameters


# Now we will define our forward propagation algorithm and cost function.

# In[ ]:


#Define our forward propogation algorithm
def forward_propagation(X,parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME') #First convolution
    A1 = tf.nn.relu(Z1) #ReLu activation on first convolution
    Z2 = tf.nn.conv2d(A1,W2,strides=[1,1,1,1],padding='SAME') #Second convolution
    A2 = tf.nn.relu(Z2) #ReLu activation on second convolution
    P = tf.contrib.layers.flatten(A2) #Flatten A2 into a vector
    Z3 = tf.contrib.layers.fully_connected(P,2,activation_fn=None) #Apply fully connected layer with two outputs
    A3 = tf.nn.sigmoid(Z3) #Sigmoid activation on fully connected layer
    
    return A3


# In[ ]:


#Define our cost function
def compute_cost(A3,Y):
    
    cost = -tf.reduce_sum(Y*tf.log(A3) + (1-Y)*tf.log(1-A3)) 
    
    return cost


# # Step 3 - Define our model
# 
# Now that our variables, forward propagation and cost function have been defined, we can bring this all together to create our convolutional neural network model.
# 
# We will start by setting up the structure of our network, and then coding a mini-batch optimization loop.  We will log the costs as this loop progresses to ensure the algorithm is behaving as intended (these costs will be plotted against iterations)
# 
# We will start by resetting out TensorFlow graph for each model call
# 
# The following steps will be taken to set up the structure of our network:
# 
# * Call and assign out placeholder values with the appropriate parameters
# 
# * Initialize our weight parameters
# 
# * Assign our forward propogation to a variable
# 
# * Assign our cost function to a variable
# 
# * Define our optimization algorithm (we will be using an Adam Optimizer for this network)
# 
# * Initialize the variables in the TensorFlow Graph
# 
# 

# In[ ]:


def model(X_train,Y_train,X_test,Y_test,learning_rate,mini_batch_size,epochs):
    costs = [] #Set up an array to store our costs at each iteration
    tf.reset_default_graph() #Reset our TensorFlow graph
    X,Y = create_placeholders(64,64,3,2) #Create our placeholder variables
    parameters = initialize_parameters() #Initialize our weight parameters
    A3 = forward_propagation(X,parameters) #Assign our forward propogation output to a variable
    cost = compute_cost(A3,Y) #Assign our cost to a varaible
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost) #Define our optimzer (Adam Optimization)
    init = tf.global_variables_initializer() #Initialize the variables in our TensorFlow graph
    temp_cost = 0
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(1,epochs + 1): #Iterate for the number of user defined epochs
            z = randint(0,X_train.shape[0] - mini_batch_size) #Chose a random integer between 0 and our sample size - user defined mini-batch size
            _ , temp_cost = sess.run([optimizer,cost],feed_dict={X: X_train[z:z+mini_batch_size], Y: Y_train[z:z+mini_batch_size]}) #Run our optimization on a randomly selected mini-batch of size 'mini_batch_size'
            #print("iteration " + str(i) +" " + str(temp_cost))
            costs.append(temp_cost)
            
        parameters = sess.run(parameters)
        print ("Parameters have been optimized.")
        
        #Plot our logged costs by iteration
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predictions_train = np.argmax(np.array(sess.run(A3,feed_dict={X:X_train,Y:Y_train})),1)
    
        accuracy_bool_train = predictions_train == np.argmax(Y_train,1)
        accuracy_perc_train = np.average(accuracy_bool_train.astype(int))
        
        predictions_test = np.argmax(np.array(sess.run(A3,feed_dict={X:X_test,Y:Y_test})),1)
    
        accuracy_bool_test = predictions_test == np.argmax(Y_test,1)
        accuracy_perc_test = np.average(accuracy_bool_test.astype(int))
        
        print('Training set accuracy is: ' + str(100*accuracy_perc_train) + "%")
        print('Test set accuracy is: ' + str(100*accuracy_perc_test) + "%")
        
        count_pos = 0
        count_true_pos = 0
        count_neg = 0
        count_true_neg = 0
        
        #The following loop will determine the accuracy of predicting pneumonia positive and negative x-ray
        for i in range(1,predictions_test.shape[0]):
            if predictions_test[i] == 1 and np.argmax(Y_test,1)[i]==1:
                count_pos = count_pos+1
            if np.argmax(Y_test,1)[i] == 1:
                count_true_pos = count_true_pos + 1
            if predictions_test[i] == 0 and np.argmax(Y_test,1)[i] == 0:
                count_neg = count_neg + 1
            if np.argmax(Y_test,1)[i] == 0:
                count_true_neg = count_true_neg + 1
                
        print("Positive pneumonia prediction accuracy: " + str((count_pos/count_true_pos)*100) + "%")
        print("Negative pneumonia prediction accuracy: " + str((count_neg/count_true_neg)*100) + "%")
    
        return accuracy_perc_train,accuracy_perc_test
            
        


# In[ ]:


tf.reset_default_graph()
model(X_train[1:3390],Y_train[1:3390],X_test,Y_test,0.000025,32,1500)


# # Step 4 - Analyzing our results
# 
# Our results are looking good.  In addition to our overall excellent training and test prediction accuracy, both our positive and negative predictions independently have beyond acceptable accuracy.
# 
# Looking at our cost vs. iterations graph, we see a general downward trend which seems to flatten out.  This suggests we have chosen an appropriate learning rate and number of iterations to balance weight optimization time and accuracy. 
# 
# We have created a convolutional neural network that can be used to predict pneumonia from X-ray images!  
# 
# This concludes the kernel.
# 
# # Thank you for viewing!

# In[ ]:




