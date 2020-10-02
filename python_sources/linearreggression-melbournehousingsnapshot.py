#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))


# In[ ]:


#import the dataset
dataset=pd.read_csv('../input/melb_data.csv')
dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset.index


# In[ ]:


dataset.columns


# In[ ]:


#we are checking the missing values
dataset.isnull().sum()


# In[ ]:


#replace the Nan values with mean
dataset['Car'].mean()


# In[ ]:


dataset['Car']=dataset['Car'].replace(np.NaN,dataset['Car'].mean())


# In[ ]:


dataset['BuildingArea']=dataset['BuildingArea'].replace(np.NaN,dataset['BuildingArea'].mean())


# In[ ]:


dataset['YearBuilt']=dataset['YearBuilt'].replace(np.NaN,dataset['YearBuilt'].mean())


# In[ ]:


dataset.isnull().sum()


# In[ ]:


ca=dataset.iloc[ : , :-1].values
ca


# In[ ]:


#'CouncilArea' is a big problem so we'll convert into numerical values
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()


# In[ ]:


ca[ : ,0]=le.fit_transform(ca[ :,0])
ca


# In[ ]:


dataset['CouncilArea']=dataset['CouncilArea'].replace(np.NaN,0)


# In[ ]:


dataset.isnull().sum() #no more missing value


# In[ ]:


#splitting the dataset into Training and Test set 
rn=dataset['Regionname'] #we are converting the region name to numeric values bec we'll draw figure of regionname and price


# In[ ]:


rn=le.fit_transform(rn)


# In[ ]:


#so firstly,we are drawing figure of dataset to understand easily dataset
plt.scatter(rn,dataset['Price'])


# In[ ]:


X=dataset['Regionname']

Y=dataset['Price']


# In[ ]:


X


# In[ ]:


rn


# In[ ]:


rn.reshape(-1, 1)


# In[ ]:


Y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test=train_test_split(rn,Y,test_size=0.2) #%80 train,%20 test.It is choosing randomly


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


X_train=X_train.reshape(-1,1)
X_train


# In[ ]:


X_test=X_test.reshape(-1,1)#we can checking to random selection
X_test


# In[ ]:


len(Y_train)


# In[ ]:


len(Y_test)


# In[ ]:


Y_test


# In[ ]:


#now,we'll do linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[ ]:


lr.fit(X_train,Y_train)


# In[ ]:


lr.predict(X_test)


# In[ ]:


Y_test


# In[ ]:


lr.score(X_test,Y_test) #Accuracy is 0.0033%

