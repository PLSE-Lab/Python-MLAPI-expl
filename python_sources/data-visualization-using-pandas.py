#!/usr/bin/env python
# coding: utf-8

# 
# # Pandas Data Visualization 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
df3 = pd.read_csv('../input/df3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df3.info()


# In[ ]:


df3.head()


# ** Recreate this scatter plot of b vs a.

# In[ ]:


df3.plot.scatter(x='a',y='b',c='red',s=50,figsize=(12,3))


# ** Create a histogram of the 'a' column.**

# In[ ]:


df3['a'].plot.hist()


# In[ ]:


plt.style.use('ggplot')


# In[ ]:


df3['a'].plot.hist(alpha=0.5,bins=25)


# ** Create a boxplot comparing the a and b columns.**

# In[ ]:


df3[['a','b']].plot.box()


# ** Create a kde plot of the 'd' column **

# In[ ]:


df3['d'].plot.kde()


# ** Figure out how to increase the linewidth and make the linestyle dashed.

# In[ ]:


df3['d'].plot.density(lw=5,ls='--')


# 

# In[ ]:


df3.iloc[0:30].plot.area(alpha=0.4)


# 
# Display the legend outside of the plot as shown below

# In[ ]:


f = plt.figure()
df3.iloc[0:30].plot.area(alpha=0.4,ax=f.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()


# # Do Up Vote
