#!/usr/bin/env python
# coding: utf-8

# ## Intro to NumPy

# In[1]:


lis = [1,2,3,4,5,6]


# In[4]:


import numpy as np


# In[3]:


np.array(lis)


# In[6]:


myMatrix = [[1,2,3],[4,5,6],[7,8,9]]


# In[7]:


np.array(myMatrix)


# In[9]:


np.arange(0,10,2)


# In[10]:


np.zeros(10)


# In[11]:


np.zeros((3,4))


# In[23]:


np.ones((3,4))


# In[15]:


np.linspace(2,10,23)


# In[17]:


np.random.rand(3)


# In[18]:


np.random.randn(3)


# randn gives points from a standard normal distribution.

# In[20]:


np.random.randn(3,3)


# In[22]:


np.random.randint(10,100,3)


# In[24]:


randarr = np.random.randint(10,100,25)


# In[25]:


randarr


# In[29]:


np.reshape(randarr,(5,5))


# In new dimension, Row * Col must be == to len(np_array)

# In[31]:


randarr.max()


# In[32]:


randarr.argmax()


# returns the index of max of array

# In[34]:


randarr.min()


# In[35]:


randarr.argmin()


# In[36]:


randarr.shape


# In[39]:


ran = randarr.reshape(5,5)


# In[40]:


ran


# In[41]:


ran.shape


# In[42]:


ran.dtype


# ### Indexing and Selection in Numpy

# Very much similar to usual lists

# In[81]:


liss = np.array(list(range(0,24,2)))
liss


# In[82]:


liss[7]


# In[83]:


liss[:]


# In[84]:


liss[5:]


# In[85]:


liss[-1:-12:-1]


# In[86]:


liss[-1:0:-1]


# Unlike normal lists, np arrays can broadcast values.

# In[88]:


liss[1:5] = 5
liss


# In[89]:


lis = liss[1:5]


# In[90]:


lis[:] = 0


# In[91]:


lis


# In[92]:


liss


# As observed, slicing does not copy the np array unlike lists as changes are visible to original array.

# In[93]:


lis[0] = 99


# In[94]:


lis


# In[95]:


liss


# lis was image of liss from index 1:5 so the lis[0] assignment in lis was refected at 1 index of liss.

# In[99]:


lis = liss.copy()
lis


# In[101]:


lis[:4] = 32
lis


# In[102]:


liss


# In[116]:


alist = np.array([list(range(1,26))])
alist = alist.reshape(5,5)
alist


# In[113]:


alist[1][1]


# In[118]:


alist[1:4,1:4]


# In[123]:


blist = np.array(list(range(2,30)))
blist


# In[127]:


temp = blist > 9
temp


# In[128]:


blist[temp]


# More direct approach to suffice above two steps would be:

# In[129]:


blist[blist > 9]


# #### Operations in NumPy

# In[10]:


alist = np.array(list(range(0,24,2)))
alist


# In[11]:


alist + alist


# In[12]:


alist - alist


# In[13]:


alist * alist


# In[14]:


alist / alist


# Notice that the division by zero in the first element is treated as a warning in NumPy and 'nan' type is provided in the output.

# In[18]:


alist ** 3


# In[19]:


alist - 100


# In[23]:


np.sqrt(alist)


# In[24]:


np.exp(alist)


# In[25]:


np.max(alist)


# In[26]:


alist.max()


# In[27]:


np.sin(alist)


# In[28]:


np.log(alist)


# Notice the -inf as the output for log 0.

# https://docs.scipy.org/doc/numpy/reference/ufuncs.html

# In[ ]:




