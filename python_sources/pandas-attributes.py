#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as  pd

d={'Name':pd.Series(['tashi','Akash','shamu']),
  'Age':pd.Series([23,43,12]),
  'Rating':pd.Series([23,54.65,34.56])}


# In[ ]:


df = pd.DataFrame(d)
print(df)


# In[ ]:


#add the dataset
print(df.sum())


# In[ ]:


#mean of dataset
print(df.mean())


# In[ ]:


print(df['Age'].mean())
#finding the mean of age column


# In[ ]:


#standard deviation
print(df.std())


# In[ ]:


#to find minimum
print(df.min())


# In[ ]:


#Maximum
print(df.max())


# In[ ]:


#to print absolute values
print(df['Age'].abs())


# In[ ]:


#to print product
print(df.prod())


# In[ ]:


#print commulative sum
print(df.cumsum())


# In[ ]:


#print commulativ product
print(df['Age'].cumprod())


# In[ ]:


#to print mode of the age
print(df['Age'].mode())


# In[ ]:




