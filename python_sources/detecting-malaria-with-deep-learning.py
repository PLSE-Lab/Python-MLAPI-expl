#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import img_to_array
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images"))


# Any results you write to the current directory are saved as output.


# **What is Malaria ? **
# Malaria is a disease transmitted from person to person by a mosquito sting, manifested by sweat caused by fever, chills and seizures.
# 

# **Reading data**

# In[ ]:


parasitized_data = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")
print("Parasitized Data =",parasitized_data[:10])
uninfected_data=os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")
print("Uninfected Data = ",uninfected_data[:10])


# **Visualized data**

# In[ ]:


for i in range(5):
    img=cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+parasitized_data[i])
    plt.imshow(img)
    plt.title("Parasitized")
    plt.show()


# In[ ]:


for i in range(5):
    img=cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+uninfected_data[i])
    plt.imshow(img)
    plt.title("Uninfected")
    plt.show()


# **One can clearly see the difference between the infected and uninfected. You can observe a small clot inside the cellular image for infected while for the uninfected the cellular image is clean without any clots.**

# **We've adjusted the picture sizes.**

# In[ ]:


data=[]
labels=[]
for i in parasitized_data:
    try:
        img=plt.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+i)
        img_resize=cv2.resize(img,(40,40))
        img_array=img_to_array(img_resize)
        data.append(img_array)
        labels.append(1)
    except:
        None
for i in uninfected_data:
    try:
        img=plt.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+i)
        img_resize=cv2.resize(img,(40,40))
        img_array=img_to_array(img_resize)
        data.append(img_array)
        labels.append(0)
    except:
        None


# In[ ]:


image_data=np.array(data)
labels=np.array(labels)


# **We shredded the data for training and testing.**

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(image_data,labels,test_size=0.14,random_state=42)


# **You must have the same data size to process it. So we adjusted the data sizes.**

# In[ ]:


x_train_shape=x_train.shape[0]
x_test_shape=x_test.shape[0]

X_train_flatten=x_train.reshape(x_train_shape,x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
X_test_flatten=x_test.reshape(x_test_shape,x_test.shape[1]*x_test.shape[2]*x_test.shape[3])


# **Transposes of the data were obtained.**

# In[ ]:


x_train=X_train_flatten.T
x_test=X_test_flatten.T
y_train=y_train.T
y_test=y_test.T


# In[ ]:


# a=size (x_train.shape[0])
def initializing_parameters(a):
    w=np.full((a,1),0.01)
    b=0.0
    return w,b
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head


# In[ ]:


def forward_propagation(w,b,x_train,y_train):
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=-(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head) # -(1-y)log(1-y^)-ylog(y^) y=y_train y^=y_head
    cost=(np.sum(loss))/x_train.shape[0]
    #Backward Propagation
    new_w=np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    new_b=np.sum(y_head-y_train)/x_train.shape[1]
    return new_w,new_b,cost


# In[ ]:


def update(w,b,x_train,y_train,learning_rate,number_iteration):
    for i in range(number_iteration):
        new_w,new_b,cost=forward_propagation(w,b,x_train,y_train)
        w=w-learning_rate*new_w
        b=b-learning_rate*new_b
        if(i%10==0):
            print("Cost After Iteration {} : {}".format(i,cost))
        parameters={"weight":w,"bias":b}
    return parameters,new_w,new_b


# In[ ]:


def predict(w,b,x_test):
    y_head=sigmoid(np.dot(w.T,x_test)+b)
    prediction=np.zeros((1,x_test.shape[1]))
    for i in range(y_head.shape[1]):
        if(y_head[0,i]>0.5):
            prediction[0,i]=1
        else:
            prediction[0,i]=0
    return prediction
        


# In[ ]:


def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,number_iteration):
    a=x_train.shape[0]
    w,b=initializing_parameters(a)
    parameters,new_w,new_b=update(w,b,x_train,y_train,learning_rate,number_iteration)
    
    prediction_test=predict(parameters["weight"],parameters["bias"],x_test)
    prediction_train=predict(parameters["weight"],parameters["bias"],x_train)
    print("train accuracy: {} ".format(100 - np.mean(np.abs(prediction_train - y_train)) * 100))
    print("test accuracy: {} ".format(100 - np.mean(np.abs(prediction_test - y_test)) * 100))
    
    


# In[ ]:


logistic_regression(x_train, x_test, y_train, y_test,learning_rate = 0.01, number_iteration = 130) 
# Number Iteration can be increased
# Learning rate can be reduced.


# **Logistic Regression with Sklearn**

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
LogisticRegression(solver='lbfgs')


# In[ ]:



logistic=LogisticRegression(random_state=42,max_iter=130)
logistic.fit(x_train.T,y_train.T)


# In[ ]:


logistic.score(x_test.T,y_test.T) # Test accuracy


# In[ ]:


logistic.score(x_train.T,y_train) # Train accuracy


# In[ ]:




