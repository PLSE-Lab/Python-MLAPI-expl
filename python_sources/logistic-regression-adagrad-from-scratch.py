#!/usr/bin/env python
# coding: utf-8

# **Content:**
# 
# ### [1. Data Preprocessing](#1)
# #### [1.1 Pulling the photos from folders with their paths](#11)
# #### [1.2 Data transportation and creating training and test sets](#12)
# #### [1.3 Creating label arrays for training and test sets](#13)
# #### [1.4 Shuffle the pics and pixels ](#14)
# ### [2. Development of a Logistic Regression Algorithm From Scratch](#2)
# ### [3. Model Performance ](#3)
# ### [4. Three line and better accuracy with sklearn library](#4)
# 

# <a id="1"></a> <br>
# ## 1. Data Preprocessing

# In[ ]:


import numpy as np
import math
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from numpy import inf
from keras import preprocessing
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
cwd = os.getcwd()
os.chdir(cwd)
print(os.listdir("../input"))


# <a id="11"></a> <br>
# ### 1.1 Pulling the photos from folders with their paths*

# * We firstly make two list with our cats and dogs images paths in order to upload them easly.

# In[ ]:


path_cats = []
train_path_cats = "../input/cat-and-dog/training_set/training_set/cats/"
for path in os.listdir(train_path_cats):
    if '.jpg' in path:
        path_cats.append(os.path.join(train_path_cats, path))
path_dogs = []
train_path_dogs = "../input/cat-and-dog/training_set/training_set/dogs/"
for path in os.listdir(train_path_dogs):
    if '.jpg' in path:
        path_dogs.append(os.path.join(train_path_dogs, path))
len(path_dogs), len(path_cats)


# * Now we have file path lists as path_dogs and path_cats
# * We will use them in a for loop to create training and test sets
# * We have total 8005 pics will seperate them into training (6000 pic.) and val (2000 pic.) sets
# * In training set (6000 pic) first 3000 pic will be dog pic and the rest will be cat pics
# * In test set (2000 pic) first 1000 pic will be dog pic and the rest will be cat pics
# * It is important to now about the arrangement of them because we need to create an label list either
# 

# <a id="12"></a> <br>
# ### 1.2 Data transportation and creating training and test sets

# In[ ]:


# load training set
training_set_orig = np.zeros((6000, 32, 32, 3), dtype='float32')
for i in range(6000):
    if i < 3000:
        path = path_dogs[i]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        training_set_orig[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i - 3000]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        training_set_orig[i] = preprocessing.image.img_to_array(img)


# In[ ]:


# load test set
test_set_orig = np.zeros((2000, 32, 32, 3), dtype='float32')
for i in range(2000):
    if i < 1000:
        path = path_dogs[i + 3000]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        test_set_orig[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i + 2000]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        test_set_orig[i] = preprocessing.image.img_to_array(img)


# * Our model will be a logistic regression model so that we need to flatten our 3d image array 
# * After reshaping we will have,
#      1. 67500 features for each picture in training set (6000 pics),
#      2. 67500 features for each picture in test set (2000 pics),  

# In[ ]:


training_set_orig.shape


# In[ ]:


x_train_ = training_set_orig.reshape(6000,-1).T
x_train_.shape


# In[ ]:


x_test_ = test_set_orig.reshape(2000,-1).T
x_test_.shape


# <a id="13"></a> <br>
# ### 1.3 Creating label arrays for training and test sets

# * Lets create target arrays for our data
#     1. We have 6000 pic in training set first 3000 is dog pic (cat label : 0) and the rest of it cat pic (cat label : 1)
#     2. We have 2000 pic in test set first 1000 is dog pic (cat label : 0) and the rest of it cat pic (cat label : 1)
#     3. Concatenating 3000 zeros and 3000 ones will give as a target array for training set

# In[ ]:


# make target tensor
y_train_ = np.zeros((3000,)) # First 3000 was dog picture so our label is 0
y_train_ = np.concatenate((y_train_, np.ones((3000,)))) # Second 3000 was cat picture so our label is 1
y_test_ = np.zeros((1000,))
y_test_ = np.concatenate((y_test_, np.ones((1000,))))
print("Training set labels" +str(y_train_.shape)+ "  Test set labels" + str(y_test_.shape))


# In[ ]:


y_train_ = y_train_.reshape(1,-1)
y_test_ = y_test_.reshape(1,-1)
print("Training set labels" +str(y_train_.shape)+ "  Test set labels" + str(y_test_.shape))


# <a id="14"></a> <br>
# ### 1.4 Shuffle the pics and pixels 

# * We need to shuffle the samples sort and features for each epoch in order to increasing the quality of training.
# * To shuffle data with row and columns I wrote a function below for two dimensional arrays.
# * With this function you can shuffle train set (x) and its label set (y) simultaneously with same new indexes.
# * As a result for example 38th pic will remain labeled as 0 (non cat) even after shuffle it.

# In[ ]:


np.arange(x_train_.shape[1])


# In[ ]:





# In[ ]:


def shuffle_xy(x,y,axis):
    """  Shuffle a two dimensional two array's contents simultaneously in accordance with axis parameter. """   

    
    if (axis == 1 or axis == "columns"):
        c = np.arange(x.shape[1])
        np.random.shuffle(c)
        shuf_x = x[:,c]
        shuf_y = y[:,c]
        shuffled = {"shuffled_x" : shuf_x,
                    "shuffled_y" : shuf_y}
        return shuffled

    
    if (axis == 0 or axis == "rows"):
        r = np.arange(x.shape[0])
        np.random.shuffle(r)
        shuf_x = x[r,:]
        shuffled = {"shuffled_x" : shuf_x}
        return shuffled
        
    else:
        print("Please write an axis argument properly.")    
        
    shuffled = {"shuffled_x" : shuf_x,
                "shuffled_y" : shuf_y}
    return shuffled
    
        


# #### 1.4.1 Shuffle the pics

# In[ ]:


shuffled_dic = shuffle_xy(x_train_,y_train_,1)
x_train_= shuffled_dic["shuffled_x"]
y_train_ = shuffled_dic["shuffled_y"]


# In[ ]:


x_test = x_test_.copy()
y_test = y_test_.copy()


# * Before training we need to normalize the data in order to 
# * When working with images usually we normalize data by dividing each pixel to 255 which is the max value of a pixel can have
# * After normalization of data we have all data between 0 and 1 
# * Why do we do that? It is about speed of training and increasment on accuracy. 
# * It should be noted that normalization is very important in order to be able to teach the relationship between the parameters we will use in training more clearly and prevent exploding or vanishing gradient.
# * If you dont know anything about exploding or vanishing gradient terms don't worry and trust me you will learn them in your machine learning voyage.
# * Urvashi Jaitley has done a good comparison study on accuracy between the use of normalized and non-normalized data in the training of the model.
# * According to that study at same epoch of training same model with normalized(NOR) and unnormalized(UNNOR) datas while NOR model has %88.93 accuracy on test set UNNOR model reach only  %48.80 accuracy value
# [https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029](http://)

# In[ ]:


x_train = x_train_/255
x_test = x_test_/255


# * By the way, I would like to remind you that the pictures work in 8-bit integers.
# * We need to use the appropriate format to visualize the image data.

# In[ ]:


index = 2000
plt.imshow(np.uint8(training_set_orig[index]))


# ## Summarize of first section (Data Prepocessing)
# 
# 1. Pull data from folders
# 2. Make a path list in order to iterate it for data tranportation
# 3. Obtain numeric values of images by keras library
# 4. Assign them to training and test sets
# 5. Create a target set includes label datas
# 6. Visualize a specific image
# 7. Normalization of data
# 8. Shuffle data and samples
# 
#  Now we have **x_train** ($67500x6000$), **y_train**($1x6000$), **x_test**($67500x2000$), **y_test**($1x2000$)

# <a id="2"></a> <br>
# ## 2. Develop a Machine Learning Model from Scratch

# In[ ]:


N = x_train.shape[0]
#W = np.full((1, x_train.shape[0]),0)
W = np.random.rand(1, x_train.shape[0])/100
grad_past = np.full((1, x_train.shape[0]),0)

b = 0
show = 1 # Python will print loss value every "show" epoch.
e = 0.000001
epoch = 400


# In[ ]:


W.shape


# In[ ]:


x_train.shape


# * Now we can obtain y with np.dot(W,x) 
# * $1$ x $3072$ * $3072$ x $6000$ = $1$ x $6000$

# * Batch normalization is using generally for if you have so many hidden layers
# * But we don't have any so we will not use below function
# * It standardize the output af an activation function

# In[ ]:


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# In[ ]:


def propagate(w_p, x_p, b_p, Y_p, alpha, grad_list, momentum, G):
    z = np.dot(w_p, x_p) + b_p

    a = sigmoid(z)
    
    loss = -(1 / N) * np.sum(Y_p * np.log(a) + (1 - Y_p) * (np.log(1 - a))) #Binary cross entropy
    
    dw = (1 / N) * np.dot((a - Y_p), x_p.T)
    #print("dw: ",dw.shape)
    db = (1 / N) * np.sum(a - Y_p)
    learning_rate, G = adagrad2(dw, G)
    W = w_p - (np.array(dw) @ learning_rate)
    #W = w_p - ((momentum * np.array(grad_past)) + (alpha * np.array(dw)))
    #print("W: ", W.shape)    
   # W = w_p - (alpha * np.array(dw))
    b = b_p - alpha*db

    new_values = {"w" : W,
             "b" : b,
             "dw" : dw,
             "cost" : loss}
    return new_values, G
    
    


# In[ ]:


def training (w_p,X_p, b_p, y_p,alpha,epoch, grad_past_p,momentum_p,show_n_epoch=1):   
    G = 0
    w = w_p[:]
    b = b_p
    x = X_p[:]
    y = y_p [:]
    cost = []
    grad_list = [[grad_past_p]]
    for i in range(epoch):
        shuffled = shuffle_xy(x, y, 1)
        x = shuffled["shuffled_x"]
        y = shuffled["shuffled_y"]
        new_values, G = propagate(w, x , b, y,alpha,grad_list, momentum_p, G)
        grad_list.append(new_values["dw"])
        w = new_values["w"]
        b = new_values["b"]        
        c= new_values["cost"]
        cost.append(c) 
        if i == 0 or i % show_n_epoch == 0:
            print("Epoch :{}  Loss: {}".format(i,c))
    grad = {"W" : w,
            "b" : b,}
            
    
    return grad,cost


# In[ ]:


def adagrad(dw,G, learning_rate = 0.005):

    dw = np.array((dw))
    dw = dw.reshape(1,-1)
    g = dw.T@dw
    zeros = np.zeros(g.shape, float)
    np.fill_diagonal(zeros, 1)
    g = g * zeros
    G = G + g
    epsilon = np.zeros(G.shape, float)
    np.fill_diagonal(epsilon, 0.000001)
    total = epsilon + G
    total_new = 1/np.sqrt(total)
    total_new[total_new == inf] = 0
    total_new = learning_rate * total_new
    return total_new, G


# In[ ]:


epsilon = np.zeros((4,4), float)
np.fill_diagonal(epsilon, 0.001)
epsilon


# In[ ]:


a = 0
epsilon + a


# In[ ]:


def prediction(w_p, x_p, b_p):
    a = sigmoid(np.dot(w_p, x_p) + b_p)
    Y = ((a >= 0.5).astype(int))
    return Y,a


# In[ ]:


def accuracy_calculator(W,X,b,y,set_name):
    y_pred = prediction(W,X,b)  
    m = y.shape[1]
    same = (y_pred == y).astype(int)
    acc = 100*(np.sum(same)/m)
    print (" From Scratch Model {} accuracy %{}".format(set_name,acc))
    return acc


# In[ ]:


def model(w_p,x_p, b_p, Y_p,alpha,epoch,x_test, y_test, grad_past, momentum, show_n_epoch):
    
    grad, cost = training(w_p,x_p, b_p, Y_p, alpha, epoch, grad_past, momentum,show_n_epoch)
    W = grad["W"]
    b = grad["b"]
    
    acc = accuracy_calculator(W,x_test,b_p,y_test,"Test") # To print test accuracy
    accuracy_calculator(W,x_p,b_p,Y_p,"Training") # To print training accuracy

    output = {"acc" : acc,
              "W"    : W,
              "b"    : b, 
              "cost" : cost}
    
    return output


# <a id="3"></a> <br>
# ## 3. Model Performance

# In[ ]:


output = model(W, x_train, b, y_train_, alpha, epoch,x_test, y_test,grad_past,momentum,show_n_epoch=10)


# In[ ]:


cost = output["cost"]
plt.plot(cost)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Loss Curve")
plt.show()


# * Although our Cost value is gradually decreasing, what we really desire is high test accuracy.
# * The obtained test accuracy is not enough to make a model successful
# * However, such models are important to see in practice the problems encountered in machine learning.
# * For example, you can get cost errors by increasing the learning rate (alpha) or by changing the initialization of w matrix
# * The reason for the error is, our cost function includes sth. like log (1- sigmoid (x)) 
# * Sigmoid function outputs 1 against high values and if we have 1 for output of sigmoid  log (1-1) will be log (0)
# * As you know, log (0) is undefined

# #### In summary, artificial neural networks are especially needed for this very reason :)
# 
# Extra knowledge: 
# 
# * Photos are compelling data due to high number of features
# * So that each pixel is a featured
# * It is a general knowledge that an efficient model is necessary if the amount of training data is 10 times the total W amount.
# * Each feature increases the number of W
# * Convolutional Neural Networks can be more efficiently modeled for problems involving photo data by keeping the number of w lower than ANN

# <a id="4"></a> <br>
# ## 4. Three line code and better accuracy with sklearn library

# In[ ]:


y_train_lib = y_train_.reshape(-1)
y_test_lib = y_test_.reshape(-1)


# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 10)
logreg.fit(x_train.T, y_train_lib.T)
print("test accuracy: {} ".format(logreg.score(x_test.T, y_test_lib.T)))
print("train accuracy: {} ".format(logreg.score(x_train.T, y_train_lib.T)))


# 
# #### Please do not hesitate to comment and ask questions.
# #### If you found it useful, I would appreciate it if you upvote

# [![smile.jpg](https://i.postimg.cc/0jVh4z64/smile.jpg)](https://postimg.cc/fS0HtTf7)
