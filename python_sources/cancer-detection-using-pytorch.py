#!/usr/bin/env python
# coding: utf-8

# This Notebook is Just a very simple implementation of Neural Network to classify Cancer Cells using pytorch
# 
# Lets first load basic libraries. We will use pandas, torch modules only.

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


df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')


# Lets have a look at Data

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# So we dont need id column and it appears there is a column which is all Nan. we will remove them by using df.drop

# In[ ]:


df.drop(['id',df.columns[-1]],axis=1,inplace=True)


# In[ ]:


df.isnull().sum()


# So now our data is clean. So we are going to build a classifier. lets convert diagonsis to category class. Category class provides us a handy function to convert data into category classes.

# In[ ]:


df['diagnosis']=df['diagnosis'].astype('category').cat.as_ordered()


# Lets check what is correaltion between 'diagonisis' column and other columns of dataframe

# In[ ]:


df.corrwith(df['diagnosis'].cat.codes)


# Here we can also drop columns with very low correlation

# Lets have a final look at df to make sure we have correct Dtypes

# In[ ]:


df.info()


# Not copy diagonisis column to a variable Y. we are coping them as classes

# In[ ]:


Y=df['diagnosis'].cat.codes


# Copy input parameters to variable X 

# In[ ]:


X=df[df.columns[1:-1]]


# lets normalize the X. we will use builtin pandas functions.

# In[ ]:


X=(X-X.mean())/X.std()


# Final look at X

# In[ ]:


X


# In[ ]:


X.shape


# But we are still working with pandas module.We would need to convert our input into torch tensors.Now lets do that

# In[ ]:


import torch as T


# pandas Series can be converted into tensor by first converting into numpy and then using torch function to_numpy

# In[ ]:


X=T.from_numpy(X.to_numpy())


# We  also need to convert X to float (it is already in float but to be sure) otherwise it will show error that mat3 need to be in float

# In[ ]:


X=X.float()


# Note: we use .copy() as series.to_numpy returns a view hence we need a copy so we dont mess up original data in some way

# In[ ]:


Y=T.from_numpy(Y.to_numpy().copy())


# Lets create Our Model

# In[ ]:


import torch.nn as nn
class Network(nn.Module):
    def __init__(self,i_dim,o_dim):
        super(Network,self).__init__()
        self.layer1=nn.Sequential(
                    nn.Linear(i_dim,128),
                    nn.ReLU(),
                    nn.Linear(128,64),
                    nn.Tanh()
        
        
        )
        self.layer2=nn.Sequential(
                    nn.Linear(64,16),
                    nn.ReLU()
                    
        )
        self.layer3=nn.Linear(16,o_dim)
        self.output=nn.LogSoftmax()
        
    def forward(self,X):
        X=self.layer1(X)
        X=self.layer2(X)
        X=self.layer3(X)
        return self.output(X)


# Lets create a instance of model

# In[ ]:


net=Network(X.shape[1],2)


# In[ ]:


print(net)


# Model is simple:
# Sequential Layer in pytorch just means applying each operation or layer in sequence. it also helps in writing forward function
# first hidden layer is Linear Layer i.e. output is calculated as Y=X*theta+bias where theta is weights. followed by a ReLu activation with 128 nodes
# 2nd layer is a linear layer followed by tanh activation. with 64 nodes
# 3rd layer is Linear layer followed by Relu layer with 16 nodes
# and output layer is Linear layer with Log softmax activation with 2 nodes.This is also our no. of classes 
# 
# 
# 

# Now to decide loss function and what optimizer to use
# 
# We can use Binary classifier or CrossEntropyLoss. We will use CrossEntropyLoss as we are treating Y as class types.
# for optimizer we will use adams optimizer

# In[ ]:


creterion1=nn.CrossEntropyLoss()
optim=T.optim.Adam(net.parameters())


# we pass paramaters of our network to adam. This can be understood as creating a link so that when we call our model and graph is formed for backprop , our optimizer will automatically upgrade the parameters for our network

# Actually we can also visualize this process

# In[ ]:


get_ipython().system('pip install torchviz')

from torchviz import make_dot


# In[ ]:


make_dot(net(X))


# So this is our flow of data. As pytorch is dynamic, There is no pre determined graph . but we can use backward graph created during each operation(due to backprop). Above graph is that
#                     

# In[ ]:


from sklearn.model_selection import train_test_split as ttt
X_train,X_test,Y_train,Y_test=ttt(X,Y,test_size=0.1)


# split the dataset for test and train

# In[ ]:


Y_train


# perform training for 1000 epochs

# In[ ]:


losses=[]
for e in range(1000):
    optim.zero_grad()
    y_hat=net.forward(X_train.float())
    loss=creterion1(y_hat,Y_train.long())
    loss.backward()
    optim.step()
    if e%10==0:
        losses.append(loss.item())
        print('loss at epoch {} is {}'.format(e,loss.item()))


# so how our loss actually looks

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(range(len(losses)),losses)


# for prediction purposes

# In[ ]:


def predict(model,X):
    y_hat=model.forward(X)
    y_hat=T.exp(y_hat)
    return y_hat.max(axis=1)


# In[ ]:


y_hat=predict(net,X_test)


# In[ ]:


y_hat


# Lets calculate accuracy of our model

# In[ ]:


acc=sum(y_hat[1]==Y_test).float()/len(y_hat[1])


# In[ ]:


acc


# So how many does we misclassified

# In[ ]:


from sklearn.metrics import confusion_matrix as cm


# In[ ]:


cm(y_hat[1].numpy(),Y_test.numpy())

