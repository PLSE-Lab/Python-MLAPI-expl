#!/usr/bin/env python
# coding: utf-8

# # Pearson correlation coefficient

# In[4]:



# import libraries
import numpy as np
import pandas as pd


# In[1]:


Iris = pd.read_csv("../input/Iris_data.csv",na_values="n/a")


# In[4]:


Iris.head()


# In[7]:


Iris_vc = Iris[Iris.Species == 'Iris-versicolor']


# In[8]:


Iris_vc


# In[9]:


Iris_versicolor_petal_length = Iris_vc.iloc[:,2]
Iris_versicolor_petal_width = Iris_vc.iloc[:,3]


# In[10]:


Iris_versicolor_petal_length = np.array(Iris_versicolor_petal_length)
Iris_versicolor_petal_width = np.array(Iris_versicolor_petal_width)


# In[11]:


np.corrcoef(Iris_versicolor_petal_length,Iris_versicolor_petal_width)


# In[12]:


def pearson_r(x,y):
    
    corr_mat = np.corrcoef(x,y)
    return corr_mat[0,1]
    


# In[13]:


r = pearson_r(Iris_versicolor_petal_length,Iris_versicolor_petal_width)


# In[14]:


print(r)


# In[ ]:




