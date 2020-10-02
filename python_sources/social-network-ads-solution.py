#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


data = pd.read_csv("../input/Social_Network_Ads.csv")


# In[6]:


data.head()


# In[7]:


data.drop("User ID",axis=1,inplace=True)


# In[8]:


x = data.iloc[:,0:4].values
y = data.iloc[:,-1].values


# In[9]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[10]:


labelEncoder_x=LabelEncoder()
x[:,0]=labelEncoder_x.fit_transform(x[:,0])


# In[11]:


onehotencoder = OneHotEncoder(categorical_features=[0])


# In[12]:


x = onehotencoder.fit_transform(x).toarray()


# In[13]:


x = np.delete(x,0,1)


# In[14]:


x


# In[ ]:





# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


s_x = StandardScaler()


# In[19]:


s_x.fit(x_train)


# In[20]:


x_train = s_x.transform(x_train)


# In[21]:


x_test = s_x.transform(x_test)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


lrc = LogisticRegression(random_state=0)


# In[24]:


lrc.fit(x_train,y_train)


# In[25]:


y_pred = lrc.predict(x_test)


# In[26]:


lrc.score(x_test,y_test)


# In[27]:


from sklearn.metrics import confusion_matrix


# In[28]:


cm = confusion_matrix(y_test,y_pred)


# In[29]:


cm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




