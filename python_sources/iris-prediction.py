#!/usr/bin/env python
# coding: utf-8

# In[5]:


# We are working on a practice problem to classify species of Iris flower 


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


# Loading important libraries and dataset

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# In[8]:


# Loading the Iris Dataset here from the Sklearn library and create a model for predicting the type of Iris flowers.

dataset= load_iris()
df= pd.concat([pd.DataFrame(data= dataset.data, columns= dataset.feature_names), pd.DataFrame(data= dataset.target, columns= ['Target'])], axis= 1)


# In[9]:


df.head()
# a simple look in to the dataset shows that there are four features (sepal and petal width and length) and a target variable


# In[10]:


print(df.Target.value_counts(dropna= False))
# the code above shows that there are three classes in the iris dataset 0, 1 and 2
dataset.target_names
# the code above shows the corresponding species of flower in the dataset


# # **Summary statistics**

# In[11]:


df.describe()


# In[12]:


# we'll check if there are missing values in the DataFrame

df.isna().sum()

# from the result we can see that there are no missing values, so we will proceed without any processing


# In[13]:


plt.figure(figsize= (24, 12))

for idx, cols in enumerate(dataset.feature_names):
    
    plt.subplot(2,2, idx+1)
    
    sns.boxplot(x= 'Target', y= cols, data= df)
    
    # the graph below shows that there is significant difference in features between the three classes
    # the plot also shows that there are a few possible outliers in that dataset but we'll leave them for now


# In[14]:


# we'll check if there is any significance in creating a new feature from multiplying sepals and petals

df['sepal area']= df['sepal length (cm)'].mul(df['sepal width (cm)'])
df['petal area']= df['petal length (cm)'].mul(df['petal width (cm)'])


# In[15]:


plt.figure(figsize= (24, 12))
for idx, cols in enumerate(['sepal area', 'petal area']):
    
    plt.subplot(1, 2, idx+1)
    
    sns.boxplot(x= 'Target', y= cols, data= df)
    
    # we can see that this feautre brings in some differentiation in the classes but we will go ahead and
    # create a model without these engineered features.
    
    


# In[16]:


X, y= df.drop(['Target','sepal area', 'petal area'], axis= 1), df.Target


# In[17]:


X.shape, y.shape
#checking the shape of the variables to see if the code worked right


# In[18]:


# splitting the dataset to train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 123, stratify= y)
# we split the dataset into 80:20 and stratify


# In[19]:


lr_clf= LogisticRegression()


# In[20]:


lr_clf.fit(X_train, y_train)


# In[21]:


lr_pred= lr_clf.predict(X_test)


# In[22]:


confusion_matrix(y_test, lr_pred)

# we see that we only had one miss classification


# In[23]:


f1_score(y_test, lr_pred, average= 'macro')

# we see that we have a good f1 score, I think we have a good model at hand.

