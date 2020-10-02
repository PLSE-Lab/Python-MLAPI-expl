#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('../input/data.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df_pre=df.drop(['id','diagnosis'],axis=1)


# In[7]:


plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
sns.countplot(x='diagnosis',data=df)


# In[8]:


df_pre.info()


# In[9]:


X=df_pre.iloc[:,0:30].values
y=df.iloc[:,1].values


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


lbe=LabelEncoder()
y=lbe.fit_transform(y)


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[16]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[17]:


model=Sequential()


# In[18]:


model.add(Dense(units=15,kernel_initializer='uniform',activation='relu',input_dim=30))


# In[19]:


model.add(Dense(units=15,kernel_initializer='uniform',activation='relu'))


# In[20]:


model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[21]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[22]:


model.fit(X_train,y_train,batch_size=10,epochs=100)


# In[23]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[24]:


cm


# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[26]:


all_preds = model.predict(X)


# In[27]:


all_preds = (all_preds > 0.5)


# In[28]:


df['prediction']=all_preds


# In[29]:


df.head()


# In[30]:


df['prediction'] = df['prediction'].apply(lambda x:'M' if x==True else 'B' )


# In[31]:


df.head()

