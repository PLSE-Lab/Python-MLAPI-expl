#!/usr/bin/env python
# coding: utf-8

# Good day kagglers,this is going to be my first ever kaggle kernel and i am pretty much excited to share some neural network basics with you,in this notebook i will show you how a very simple neural network can learn x to y variable mapping(this notebook is for deep learning or neural network beginners).

# **importing the necessary libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# let's define x and y variable for our neural network to learn,the cell below is containing 1-d array x with some numbers and corresponding y value is stored in 1-d array y which denoting our dependent variable

# In[ ]:


x = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)
y = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype = float)


# now you might have already observed the pattern between x to y relaion which is : if x is 1 then y is also 1 but if x is 0 then y is 0-1 = -1,when x is -1.0,y is -1.0-2.0 = -3.0,again if we go in the positive direction we can see that for x==1 y is also equal 1,for x = 2,y =2+1 =3,for x = 3,y = 3+2 = 5 and for x = 4,y = 4+3 = 7, now if x = 5 then what should be the y value for this equation? 5+4 = 9 right? similarly if x is equal to -2 then y should be -2-3 = -5 right?now lets visualize this simple mathematical function only for our defined x and y variable

# In[ ]:


plt.plot(x,y)
plt.title('Visualizing relation between x and y')
plt.ylabel('Dependent variable (y)')
plt.xlabel('Independent variable (x)')
plt.show()


# **lets define our simple neural network with single unit (because we have a simple 1-d array x),we will use sgd optimizer and mean_squared_error function for computing the loss function**
# 
# 1.to understand keras optimizer [click here](https://keras.io/optimizers/)
# 
# 2.[keras losses](https://keras.io/losses/)
# 

# In[ ]:


model = keras.Sequential([keras.layers.Dense(units =1, input_shape=[1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')


# telling our neural network(defined above) to learn x to y mapping for 100 epochs.An epoch is a single step in training a neural network. in other words when a neural network is trained on every training samples only in one pass we say that one epoch is finished.One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from underfitting to optimal to overfitting curve.to learn more [watch this](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

# In[ ]:


model.fit(x,y, epochs = 100)


# **asking our network to predict what should be the corresponding y value of given x input as 10.0 which is a new x input to our learnt neural network **

# In[ ]:


print("predicted Answer : " ,  model.predict([10.0]))


# **the network predicts that for x=10.0 ,y should be 18.098911**
# 
# now lets see what the correct answer is,
# 
# for x = 4,y was 4+3 = 7,
# 
# so,
# 
# for x = 5,y should be 5+4 = 9
# 
# for x = 6, y should be 6+5 = 11
# 
# for x = 7,y should be 7+6 = 13
# 
# for x = 8, y should be 8+7 = 15
# 
# for x = 9, y should be 9+8 = 17
# 
# for x = 10,y should be 10+9 = 19(where our neural network predicting 18.098911 which is pretty close to correct answer 19 right?)
# 

# ***Note that Our Network only learnt for 6 samples so if you ask it to predict what should be the answer of very large input like 10000000 then the network will not do well because the value you want to predict is too far away from our learnt samples,to fix this issue you will need to feed this network much more samples to learn***

# **let's try again**
# 
# remember the old school lesson for converting Celsius to Fahrenheit?
# 
# using tensorflow we will do the same now
# 
# our network will learn **Approximate Formula which is : f = c * 1.8 + 32** to predict Fahrenheit for given Celsius as input

# **Setting up training data**

# In[ ]:


celsius_q = np.array([-40,-10,0,8,15,22,38], dtype = float) #x variables

fahrenheit_a = np.array([-40,14,32,46,59,72,100], dtype = float) #y variables

#lets print these

for f,c in enumerate (celsius_q):
  print("{} degree cesius = {} degree fahrenheit".format(c,fahrenheit_a[f]))


# **Creating the model**

# In[ ]:


ten = tf.keras.layers.Dense(units = 1, input_shape = [1])

model = tf.keras.Sequential([ten])


# In[ ]:


model.compile(loss = 'mean_squared_error',
             
              optimizer = tf.keras.optimizers.Adam(0.1) #0.1 here is the learning rate
             )


# **Training the Model**

# In[ ]:


train = model.fit(celsius_q, fahrenheit_a, epochs = 100, verbose = False )

print("training has been finished")


# **Displaying Training Statistics**

# In[ ]:


plt.xlabel('Epoch Number')

plt.ylabel('Loss Magnitude')

plt.plot(train.history['loss'])


# **MAKING PREDICTION**

# In[ ]:


print(model.predict([100.0]))


# **Looking at the layer weights**

# In[ ]:


print("These are the layer variables: {}".format(ten.get_weights()))


# *i hope you enjoyed it,thanks for your time,in the upcoming kernel i will be working on CNN's in shaa allah*

# **References : **
# 
# 1. https://www.coursera.org/learn/introduction-tensorflow#utm_source=email&utm_medium=dl.aiGeneralListCTA&utm_campaign=TFSC1Announcement
# 
# 2. https://classroom.udacity.com/courses/ud187
