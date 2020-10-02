#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer = load_breast_cancer()


# In[5]:


cancer.keys()


# In[6]:


df = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])


# In[7]:


df.head(2)


# In[9]:


#apply feature scalling so that all the feature values come in the equal range
from sklearn.preprocessing import StandardScaler


# In[10]:


scalar = StandardScaler()


# In[11]:


scalled_Feature = scalar.fit_transform(df)


# In[23]:


#Apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)


# In[24]:


pca.fit(scalled_Feature)


# In[26]:


X_PCA = pca.transform(scalled_Feature)


# In[30]:


plt.scatter(X_PCA[:,0], X_PCA[:,1], c=cancer['target'], cmap='rainbow')
plt.xlabel = 'First Principle Component'
plt.ylabel = 'Second Principle Component'


# In[31]:


#Apply logistic regression on it
from sklearn.linear_model import LogisticRegression


# In[32]:


model = LogisticRegression()

#Formulate X and y
X = X_PCA
y = cancer['target']


# In[33]:


#split in data in train and test set
from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[35]:


model.fit(X_train, y_train)


# In[40]:


predictions = model.predict(X_test)


# In[41]:


#print confussion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix


# In[43]:


print(confusion_matrix(y_pred=predictions, y_true=y_test))
print('\n')
print(classification_report(y_pred=predictions, y_true=y_test))


# In[ ]:




