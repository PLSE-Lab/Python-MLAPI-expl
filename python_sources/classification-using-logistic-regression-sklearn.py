#!/usr/bin/env python
# coding: utf-8

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


import numpy as np 
import matplotlib.pyplot as plt  
import h5py 
from PIL import Image 


# This function loads the datast

# In[ ]:


#Loading the Dataset
#Code from another kernel
def load_dataset():
    train_dataset = h5py.File('../input/cat-images-dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_dataset = h5py.File('../input/cat-images-dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
X_train_org, Y_train_org, X_test_org, Y_test_org, classes = load_dataset()


# we would like to save our original datas So we can compare result better at end
# We use shape attribute to find out shape of our train and test dataset

# In[ ]:


#Code Starts Here
X_train, Y_train, X_test, Y_test=X_train_org, Y_train_org, X_test_org, Y_test_org
for x in [X_train, Y_train, X_test, Y_test, classes]:
    print(x.shape)


# We have 209 train cases i.e. 209 pictures intotal, each row of array represents  a picture
# we would like to create a feature columns for each row hence We will reshape X_train and X_test to size (64x64x3,total number of samples)

# In[ ]:


X_train=X_train.reshape([X_train.shape[0],-1])
X_test=X_test.reshape([X_test.shape[0],-1])
X_train.shape


# Now we would flatten the inputs. This brings the input values in range of 0-1

# In[ ]:


X_train=X_train/255
X_test=X_test/255


# First Lets Use Well Implemented Function in SkLearn for our logistic regression. We shall use LogistricRegression and Then SGDclassifier to compare various score with various learning rates

# In[ ]:


from sklearn.linear_model import LogisticRegression


# We will use n_jobs=-1 so to utilize every available cpu

# In[ ]:


model=LogisticRegression(n_jobs=-1)


# In[ ]:


model


# fit the model

# In[ ]:


model.fit(X_train,Y_train.T)


# In[ ]:


model.score(X_train,Y_train.T)


# In[ ]:


model.score(X_test,Y_test.T)


# SO we Have a test accuracy of 72% which is pretty nice

# Let Us try different learning rate. Logistic regression do not provide mehtod to adjust learning rate. But SGDClassifier does, so we will use it for further analysis
# 

# In[ ]:


from sklearn.linear_model import SGDClassifier


# In[ ]:


get_ipython().run_line_magic('pinfo', 'SGDClassifier')


# In[ ]:


score=[]
lr=[0.001,0.003,0.01,0.03,0.1,0.3,0.9,1]


# setting random_state=2 helps in keeping outcomes consistent.eta0 represents intial learning rate

# In[ ]:


for l in lr:
    model=SGDClassifier(loss='log',n_jobs=-1,learning_rate='constant',eta0=l,random_state=2)
    model.fit(X_train,Y_train.T)
    score.append(model.score(X_test,Y_test.T))
    


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(lr,score)


# In[ ]:


score


# Now Ones we have trained our model, we will use them to classify cat or no cat

# In[ ]:


def cat_or_no_cat(i):
    if i==1:
        return 'cat'
    else:
        return 'no cat'


# In[ ]:


def predict(X_org,model):
    X=X_org.reshape(1,-1)
    X=X/255
    Y_hat=model.predict(X)
    return Y_hat


# In[ ]:


import random
get_ipython().run_line_magic('pinfo', 'random.randint')


# To determine a cat or no cat we will use our test cases images. These images can be accesed using num variable. By default it is set to a random number in each iteration

# In[ ]:


num=random.randint(0,X_test_org.shape[0]-1)
print("IT IS A ",cat_or_no_cat(Y_test_org[0,num]))
print("According to model it IS A " ,cat_or_no_cat(predict(X_test_org[num],model)))
plt.imshow(X_test_org[num])


# This just plot accuracy over 100 data examples

# In[ ]:


acc=0
for i in range(100):
    num=random.randint(0,X_test_org.shape[0]-1)
    
    acc=acc+abs(Y_test_org[0,num]-predict(X_test_org[num],model))
print(acc/100)

