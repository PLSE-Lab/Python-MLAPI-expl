#!/usr/bin/env python
# coding: utf-8

# Here we will see a demonstration of deep Learning.If you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Importing Python Modules**

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles


# **Creating Matrix of Features:**
# 
# Here We will be creating the Matrix of Feaure X and the target value y. X will be points in the form of circle which wwill be seen clearly in the vizualization shown below.Our Task will be to Predict the correct class for y using deep learning.
# 

# In[ ]:


X,y=make_circles(n_samples=1000,noise=0.1,factor=0.2,random_state=0)


# In[ ]:


X
X.shape


# So we have 1000 rows of data and two columns columns of data.

# **Plotting the Matrix of Feature X for the two classes of y**

# In[ ]:


plt.figure(figsize=(5,5))
plt.plot(X[y==0,0],X[y==0,1],'ob',alpha=0.5)
plt.plot(X[y==1,0],X[y==1,1],'xr',alpha=0.5)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.legend(['0','1'])
plt.title('Blue circles and Red crosses')


# The above graph is the representation of our data.So from the above figure we can see that we have tow catogeries of Data represented by two distinct circles Blue and red in color.Now we will be building a machine learning model to separate/Predict the two classes classes in our dataset. 

# **Deep Learning Model Build**
# 
# Here we will be building a Neural Network with One Hidden Layer with 4 Neurons,2 Inouts and 1 output Neuron.For the hidden Layer we will use the tanh activation function and for the output layer we will be using Sigmoid Function.We will be using binary cross entropy loss function and accuracy as the mesure of model performance.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import SGD


# In[ ]:


model=Sequential()


# In[ ]:


model.add(Dense(4,input_shape=(2,),activation='tanh'))


# In[ ]:


model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.compile(SGD(lr=0.5),'binary_crossentropy',metrics=['accuracy'])


# In[ ]:


model.fit(X,y,epochs=20)


# **Defining the Decision Grid Of the model**
# 
# Wo will be plotting a graph to show that our Neural network was able properly classify our data. The White region between the red and blue displays the area of seperation between the two set of classes in our dataset. 

# In[ ]:


hticks=np.linspace(-1.5,1.5,101)
vticks=np.linspace(-1.5,1.5,101)
aa,bb=np.meshgrid(hticks,vticks)
ab=np.c_[aa.ravel(),bb.ravel()]
c=model.predict(ab)
cc=c.reshape(aa.shape)


# In[ ]:


plt.figure(figsize=(5,5))
plt.contourf(aa,bb,cc,cmap='bwr',alpha=0.2)
plt.plot(X[y==0,0],X[y==0,1],'ob',alpha=0.5)
plt.plot(X[y==1,0],X[y==1,1],'xr',alpha=0.5)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.legend(['0','1'])
plt.title('Blue circles and Red crosses')


# So the area of seperation between the two classes is shown by the white region which has a shape of a traingle.This shows our Neural Network is able to classify our data correctly.
