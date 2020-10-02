#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# let's start ...
# 
#     first we input our data... by using (pd.read_csv("__"))

# In[ ]:


data = pd.read_csv("../input/data.csv")
data


#     Now we want find the mean, standard deviation, 1st,2nd,3rd quartile, and more other information...

# In[ ]:


data.describe()


#     now we want to check is there any null value present in data in form of heapmap

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(data.isnull(),cbar=False)


#     the correlation between column atributes

# In[ ]:


data.corr()


#     to show the co-relation between diffrent column atributes in form of heatmap

# In[ ]:


plt.figure(figsize=(25,25))
sns.heatmap(data.corr())


#     By using .countplot() we want to find out the count of Age

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(x='Age',data=data,palette='rainbow')


#     now from this we find that the relation between age and stamina as well as we find the count plot of age and stamina

# In[ ]:



sns.jointplot(x='Age',y='Stamina',data=data)


#     By using .violinplot() we find the relation between BallControl and Acceleration... from this graph we find that Acceleration increases with increases of Ball control (as from density)

# In[ ]:


plt.figure(figsize=(15,10))
sns.violinplot('BallControl','Acceleration',data=data)


# In[ ]:





# In[ ]:




