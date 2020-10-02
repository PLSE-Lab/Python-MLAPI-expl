#!/usr/bin/env python
# coding: utf-8

# https://github.com/connectaditya
# # Seaborn Exercises - 
# 
# 

# ## The Data
# 
# We will be working with a famous titanic data set for these exercises. 
# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


titanic = sns.load_dataset('titanic')


# In[ ]:


titanic.head()


# # Exercises
# 
# ** Recreate the plots below using the titanic dataframe. There are very few hints since most of the plots can be done with just one or two lines of code and a hint would basically give away the solution. Keep careful attention to the x and y labels for hints.**
# 
# ** *Note! In order to not lose the plot image, make sure you don't code in the cell that is directly above the plot, there is an extra cell above that one which won't overwrite that plot!* **

# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[ ]:


sns.jointplot(x='fare',y='age',data=titanic)


# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[ ]:


sns.distplot(titanic['fare'],bins=30,kde=False,color='red')


# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[ ]:


sns.boxplot(x='class',y='age',data=titanic,palette='rainbow')


# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[ ]:


sns.swarmplot(x='class',y='age',data=titanic,palette='Set2')


# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[ ]:


sns.countplot(x='sex',data=titanic)


# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[ ]:


sns.heatmap(titanic.corr(),cmap='coolwarm')
plt.title('titanic.corr()')


# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[ ]:


g = sns.FacetGrid(data=titanic,col='sex')
g.map(plt.hist,'age')


# # Great Job!
# 
# 

# In[ ]:




