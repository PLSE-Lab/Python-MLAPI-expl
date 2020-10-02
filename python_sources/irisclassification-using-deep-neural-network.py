#!/usr/bin/env python
# coding: utf-8

# <h1>  Most basic deep learning tutorial for Beginners </h1>

# ***In this tutorial we will try to understand that how a Artifical Neural Network model works***

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


iris=pd.read_csv("/kaggle/input/iris/Iris.csv")
#or you can also write this as
#iris=pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


iris.head()


# In[ ]:


iris.info()


# In[ ]:


iris.describe()


# In[ ]:


print(iris["Species"].value_counts())


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Let's do some visulization

# In[ ]:


sns.pairplot(iris,hue="Species")
plt.show()


# <h4> this below line is not a part of our model but this line helps a lot to understad which type of parameters we can pass to a function. Some times we don't remeber the actual parameter of a function, in hat situation it can helps you alot</h4>

# In[ ]:


get_ipython().run_line_magic('pinfo', 'sns.pairplot')


# <h1> Now let's create a very simple ANN model<h1>

# In[ ]:


import tensorflow


# In[ ]:


print(tensorflow.__version__) #you can use this ".__version__" function with any library to check it's version.


# In[ ]:


from pandas import get_dummies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# <h3> Sepal length,Sepal Width,Petal Length,Petal Width are our features </h3>
#     
# <h3> Species is our training labels </h3>

# In[ ]:


feature=iris.drop(columns=["Id","Species"])
label=get_dummies(iris["Species"]) #get_dummies is used to convert your categorical values into nueric binary format


# In[ ]:


label.head()


# In[ ]:


from tensorflow.keras.optimizers import Adam
#optimizers are used to optimize or minimize your loss or error function.


# In[ ]:


sq=Sequential()
sq.add(Dense(units=10,activation="relu",input_shape=(4,))) 
#units is analogus for neurons. No of neurons in a layer is equals to units defined in your function. It is also a
#hyperperameter.
sq.add(Dense(32,activation="relu"))
#Activation functions are used to make some adjusts in your outputs.
#There are many activation function. ReLu is one of them.You can also use sigmoid,softmax,ELu,LRelu etc.
sq.add(Dense(3,activation="softmax"))
#Softmax is used to get probabilistics optput instead of a specific label output. It provides probability to each 
#and every input labels. The category which will have highest probability will be considered as output for respective
#observation.
sq.compile(loss="categorical_crossentropy",
          optimizer=Adam(lr=0.1),
          metrics=["accuracy"]
          )
#As we have three categories in input labels that why we used categorical_crossentropy.
#in metrics paramter : List of metrics to be evaluated by the model during training and testing


# In[ ]:


sq.summary()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'tensorflow.keras.models.Sequential.compile')


# In[ ]:


his=sq.fit(feature,label,epochs=100,batch_size=50,verbose=True)
#epochs is number of times a complete dataset will pass from each layer. It is also a hypermeter. 
#batch_size is no of samples that will pass through a neuron at a time.
#There is a another term used in neural network is called as "iteration". Iteratoin is = total_no_of_sample/batch_size


# <h1>Now let's visulalize how the loss and accuracy varied in each and every epoch</h1>

# In[ ]:


type(his) #The "his" object that we created above is used to keep history of our ANN model


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(his.epoch,his.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs V/S Loss")
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(his.epoch,his.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Epochs V/S Accuracy")
plt.show()


# # Thank You.
