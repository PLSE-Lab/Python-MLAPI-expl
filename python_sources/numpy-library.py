#!/usr/bin/env python
# coding: utf-8

# 
# # Numpy Libaray                     
#                                                                                          # code written by
#                                                                                             - rashiddaha

# In[ ]:


import numpy as np


# In[ ]:


#list 
list1=[1,2,3,4,5]


# In[ ]:


list1


# In[ ]:


list2=[6,7,8,9,10]


# In[ ]:


#using numpy creating mutidimensional array 
add= np.array([list1,list2])


# In[ ]:


add


# In[ ]:


add.shape


# In[ ]:


type(add)


# In[ ]:


#Re-shaping the structure of 2d array  into 2 columns and 5 arrays .
reshape_now=add.reshape(5,2)


# In[ ]:


reshape_now


# In[ ]:


#accessing the fifth Eelement
add[1]
reshape_now[4]


# In[ ]:


reshape_now[:,:]


# In[ ]:


reshape_now[:]


# In[ ]:


# creating a new list using numpy
new_list1=np.arange(0,10)


# In[ ]:


new_list1


# In[ ]:


# creating a new list using numpy
new_list2=np.arange(0,10, step=3)


# In[ ]:


new_list2


# In[ ]:


#it will repaete all elemsts from the selection element to onwords .
new_list2[2:]=40


# In[ ]:


new_list2


# In[ ]:


# generating random values in specific range
np.linspace(1,5,30)


# In[ ]:


np.arange(0,10).reshape(2,5)


# In[ ]:


np.ones((2,5),dtype=int)


# In[ ]:


#generating random 2 rows of 5 columns
np.random.randn(2,5)


# In[ ]:





# In[ ]:




