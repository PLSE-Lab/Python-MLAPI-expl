#!/usr/bin/env python
# coding: utf-8

# ### Contents
# 
# **Analysis**
# 
# **Preprocessing**
# 
# **Creating Random Forest Classifier**
# 
# **Evaluating Training and Testing set**
# 
# **Creating Support Vector Classifier**
# 
# **Evaluating Training and Testing set**

# In[1]:


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


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset = pd.read_csv('../input/weatherAUS.csv')


# In[4]:


dataset.head()


# In[5]:


dataset.isnull().sum()


# In[6]:


dataset.shape


# In[7]:


dataset.drop(labels = ['Date','Location','Evaporation','Sunshine','Cloud3pm','Cloud9am','RISK_MM'],axis = 1,inplace = True)


# In[8]:


dataset.head()


# In[9]:


dataset['RainToday'].replace({'No':0,'Yes':1},inplace = True)
dataset['RainTomorrow'].replace({'No':0,'Yes':1},inplace = True)
dataset.shape


# In[10]:


dataset.dropna(inplace = True)


# In[11]:


dataset.shape


# In[12]:


categorical = ['WindGustDir','WindDir9am','WindDir3pm']


# In[13]:


dataset = pd.get_dummies(dataset,columns = categorical,drop_first=True)


# In[14]:


dataset.head()


# In[15]:


dataset.shape


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


sc = StandardScaler()


# In[18]:


x = dataset.drop(labels = ['RainTomorrow'],axis = 1)


# In[47]:


x.shape


# In[19]:


y = dataset['RainTomorrow']


# In[20]:


x = sc.fit_transform(x)


# In[21]:


x


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 40)


# ## Random Forest Classifier

# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[24]:


rc = RandomForestClassifier(n_estimators = 200,max_leaf_nodes = 1000)
rc.fit(x_train,y_train)


# In[25]:


y_pred = rc.predict(x_test)


# In[26]:


y_train_pred = rc.predict(x_train)


# In[27]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# ## Training set evaluation

# In[28]:


print(classification_report(y_train,y_train_pred))


# In[29]:


confusion_matrix(y_train,y_train_pred)


# ## Testing set evaluation

# In[30]:


print(classification_report(y_test,y_pred))


# In[31]:


print(confusion_matrix(y_test,y_pred))


# In[32]:


print('Training accuracy ---->',accuracy_score(y_train,y_train_pred))
print('Testing accuracy  ---->',accuracy_score(y_test,y_pred))


# ## Support Vector Classification

# In[33]:


from sklearn.svm import SVC


# In[34]:


svc = SVC()


# In[35]:


svc.fit(x_train,y_train)


# In[36]:


y_pred = svc.predict(x_test)
y_train_pred = svc.predict(x_train)


# ## Training Set Evaluation

# In[37]:


print(classification_report(y_train,y_train_pred))


# In[38]:


confusion_matrix(y_train,y_train_pred)


# ## Testing Set Evaluation

# In[39]:


print(classification_report(y_test,y_pred))


# In[40]:


confusion_matrix(y_test,y_pred)


# In[41]:


print('Training Accuracy ---->',accuracy_score(y_train,y_train_pred))
print('Testing Accuracy  ---->',accuracy_score(y_test,y_pred))


# ## ANN

# In[44]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[45]:


classifier = Sequential()


# In[51]:


classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu',input_dim = 58))
classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))
classifier.add(Dense(units = 30,kernel_initializer='uniform',activation = 'relu'))
classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer='uniform'))


# In[52]:


classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


# In[54]:


classifier.fit(x_train,y_train,epochs = 50,batch_size=10)


# In[68]:


y_pred = classifier.predict_classes(x_test)
y_train_pred = classifier.predict_classes(x_train)


# In[69]:


print('Training Accuracy ---->',accuracy_score(y_train,y_train_pred))
print('Testing Accuracy  ---->',accuracy_score(y_test,y_pred))


# In[ ]:




