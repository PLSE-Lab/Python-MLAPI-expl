#!/usr/bin/env python
# coding: utf-8

# # Functions

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


def add_two(x):
    '''This function adds 2 to every element on a list'''
    results = [] # Create empty list
    for i in x:
        a = i + 2
        results.append(a)
    return results


# In[ ]:


results = add_two(list(range(1,5)))


# In[ ]:


print(results)


# # Pandas

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/sample.csv',sep=',')


# In[ ]:


df.head(20)


# ### List unique values

# In[ ]:


list(df['Target Name'].unique())


# In[ ]:


list(df['Molecule ChEMBL ID'].unique())


# ### Groupby

# In[ ]:


df.groupby('Target Name')['Molecule ChEMBL ID'].count()


# ### Plotting stuff

# In[ ]:


df.hist('Molecular Weight')
plt.show()


# In[ ]:


df.hist('pChEMBL Value')
plt.show()


# In[ ]:


sns.pairplot(df)


# # Numpy

# In[ ]:


import numpy as np


# In[ ]:


myarray = np.asarray([1,2,3,4,5]) # 1D


# In[ ]:


myarray


# In[ ]:


myarray2 = np.asarray([[1,2,3,4,5]]) # 2D


# In[ ]:


myarray3 = np.asarray([[[1,2,3,4,5]]]) # 3D


# In[ ]:


myarray2


# In[ ]:


print(myarray.shape,myarray2.shape,myarray3.shape)


# ### Slicing and indexing

# In[ ]:


bigger_array = np.random.rand(4,3)


# In[ ]:


bigger_array


# In[ ]:


bigger_array[0,0]


# In[ ]:


bigger_array[0]


# In[ ]:


bigger_array[0,:]


# In[ ]:


bigger_array[0,:bigger_array.shape[1]]


# In[ ]:


bigger_array[1,1]


# ### Math with array

# In[ ]:


mylist = [1,2,3,4,5]
for i in mylist:
    print(i+2)


# In[ ]:


myarray


# In[ ]:


add_two_array = myarray + 2
print(add_two_array)


# In[ ]:


mult_two_array = myarray * 2
print(mult_two_array)


# ### Matrix multiplication

# In[ ]:


a = np.random.rand(3,5)
b = np.random.rand(5,10)


# In[ ]:


a


# In[ ]:


b


# In[ ]:


c = a@b


# c.shape

# **REMEMBER**
# 
# A = (M,N)
# 
# B = (N,P)
# 
# C = A * B
# 
# C = (M,P)

# ### Look at home

# Statsmodel
# 
# Scipy
# 
# Source of pandas

# ### Integration of pandas with numpy

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.values


# In[ ]:




