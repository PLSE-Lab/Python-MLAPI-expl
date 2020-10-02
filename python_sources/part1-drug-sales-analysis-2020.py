#!/usr/bin/env python
# coding: utf-8

# The Dataset given in the Hackerearth Competition is based on a Drug manufacturing company that made its datasets open-sourced to calculate the base_score from the independent features provided! You will have to train your regression model with training data provided and predict base_score on given test.csv data. Here is some cruchy insights that i gained in the train.csv dataset.
# Do UPVOTE, if you LIKE the kernel.

# # Import Neccesary Libraries and Packages

# In[ ]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Import the dataset to memory

# In[ ]:


train = pd.read_csv('/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/train.csv')
test=pd.read_csv('/kaggle/input/hackerearth-effectiveness-of-std-drugs/dataset/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train=train.drop(['patient_id'],axis=1)
test =test.drop(['patient_id'],axis=1)
print(train.shape,test.shape)


# # Check for the best sold drugs along with its numbers 

# In[ ]:


x = train['name_of_drug'].value_counts()
x


# # Check if there are any drugs that has been sold more than 100 items.

# In[ ]:


import seaborn as sns
sns.countplot(train['name_of_drug'].value_counts()>100)


# # Check the effective rating given by customers for products they bought 

# In[ ]:


sns.distplot(train['effectiveness_rating'])


# # Analyse the ratings of customers and their frequency 

# In[ ]:


a = train['effectiveness_rating'].value_counts()
print(a)
sns.countplot(train['effectiveness_rating'])


# # Graph showing Drug effectiveness vs Number of times prescribed to have good idea of how doctors are prescribing drugs based on their effectiveness ratings

# In[ ]:


sns.jointplot(x='effectiveness_rating',y='number_of_times_prescribed',data=train)


# # Check highest sold drugs till date

# In[ ]:


train['name_of_drug'].value_counts().head(30).plot(kind='barh',figsize=(20,10))


# # Reason for the highest usage of drugs that states customer problem

# In[ ]:


import matplotlib.pyplot as plt
train['use_case_for_drug'].value_counts().head(40).plot(kind='barh',figsize=(10,10))


# In[ ]:


a =train['number_of_times_prescribed']
print('The mean value of number of prescribed drugs is:',a.mean())


# In[ ]:


print('The maximum number of prescribed drugs is:',a.max())
print('***********************************************')
print('The maximum number of prescribed drugs is:',a.min())


# In[ ]:


plt.plot(a)


# # Graph showing Number of prescription to total no: of sales

# In[ ]:


x = train['name_of_drug'].value_counts().head(30)
y = train['number_of_times_prescribed'].head(30)
plt.plot(x,y)
plt.show()


# # Let's check how Numerical features contribute to our target variable

# In[ ]:


corrmat = train.corr()
top=corrmat.index
plt.figure(figsize=(10,10))
graph = sns.heatmap(train[top].corr(),annot=True,cmap="Blues")


# # Do UPVOTE! If you like this kernel and show your support!
