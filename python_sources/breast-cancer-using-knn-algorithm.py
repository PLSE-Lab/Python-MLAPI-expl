#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib is used for plot the graphs,
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


data=pd.read_csv("../input/breast-cancer.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data=data.drop(["Unnamed: 32","id"],axis=1)


# In[7]:


data.head()


# In[8]:


data.columns


# In[9]:


data["diagnosis"]=data["diagnosis"].map({'B':0,'M':1}).astype(int)
data.head()


# In[10]:


corr=data.corr()
corr.nlargest(30,'diagnosis')['diagnosis']


# In[11]:


x=data[['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean','radius_se','perimeter_se', 'area_se','compactness_se', 'concave points_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','texture_worst','area_worst']]
y=data[['diagnosis']]


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[14]:


from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy_score(predict,y_test)
print(accuracy_score)


# In[15]:


accuracy=model.score(x_train,y_train)
print(accuracy)


# In[16]:


model.score(x_train,y_train)


# In[17]:


predict.max()


# In[18]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=2)
score


# In[19]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[20]:


model=RandomForestClassifier(max_depth=6,random_state=5)
model.fit(x_train,y_train)
predict=model.predict(x_test)
acc = model.score(x_test,y_test)
acc


# In[ ]:





# In[ ]:




