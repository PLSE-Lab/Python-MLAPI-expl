#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_set = pd.read_csv('../input/train.csv')


# In[3]:


train_set.shape


# In[4]:


train_set.head()


# In[5]:


train_set.columns


# ##### Lets look at missing values

# In[6]:


train_set.isnull().sum()


# Awesome, there are no missing values. what a cleaned dataset!

# In[7]:


train_set.describe()


# All the values look good, no negatives in ram,battery_power etc..<br>
# Also look at max values, front_camera is 19 and ram is 3998 which are reasonable values.<br>
# I don't see any data errors.

# In[8]:


train_set.describe(include=['O'])


# In[9]:


train_set['touch_screen'].value_counts()


# In[10]:


train_set['bluetooth'].value_counts()


# In[11]:


train_set['dual_sim'].value_counts()


# In[12]:


train_set['wifi'].value_counts()


# In[13]:


train_set['4g'].value_counts()


# In[14]:


train_set['3g'].value_counts()


# In[15]:


train_set['price_range'].value_counts()


# In[16]:


#Lets convert these categorical values into numeric
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    train_set[i] = train_set[i].replace({'yes':1,'no':0})


# In[17]:


train_set['price_range'] = train_set['price_range'].replace({'very low':0,'low':1,'medium':2,'high':3})


# ### Modeling

# ##### There are 4 distinct values in price_range. Lets start with a Decision tree classifier

# In[18]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


# In[19]:


features = list(set(train_set.columns) - set(['Id','price_range']))


# In[20]:


features


# In[21]:


X = train_set[features]
Y = train_set['price_range']


# In[22]:


trainX, testX, trainY, testY =  train_test_split(X, Y, test_size = .3)


# In[23]:


dt = DecisionTreeClassifier()
model = dt.fit(trainX,trainY)
preds = model.predict(testX)


# In[24]:


accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)


# In[25]:


print (classification_report(testY,preds))


# ###### 81% accuracy?  Good, Lets go ahead and submit and think of next steps to improve accuracy

# ---

# ### Scoring

# In[26]:


test_set = pd.read_csv('../input/test.csv')


# In[27]:


l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    test_set[i] = test_set[i].replace({'yes':1,'no':0})


# In[28]:


test_set['price_range'] = model.predict(test_set[features])


# In[29]:


test_set['price_range'] = test_set['price_range'].replace({0:'very low',1:'low',2:'medium',3:'high'})


# In[30]:


#test_set[['Id','price_range']].to_csv('1st_sub.csv',index=False)


# In[ ]:




