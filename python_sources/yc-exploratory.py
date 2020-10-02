#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import datetime 

# plotting libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('fivethirtyeight') #plot style used by fivethirtyeight
mpl.rcParams['figure.figsize'] = (12.0, 7.0)


# In[ ]:


df=pd.read_csv('../input/companies.csv')


# In[ ]:


df.head()


# In[ ]:


df.loc[df['year'] == 2005, ['name']]


# In[ ]:


df['vertical', 'year','batch']


# In[ ]:


df['vertical', 'year','batch']


# In[ ]:


print ("The total number of companies funded by YC since 2005:", df.shape[0])


# The dataset has very few columns so we don't have that many variables to observe.

# ## The number of companies funded per year

# One interesting question is the number of companies that YC has funded each year of it's existence

# In[ ]:


sns.countplot(df.year)
plt.title('# of companies funded per year')
plt.ylabel('Number')


# Clearly we see that every year, there is an increase in the number of companies funded by YC. This is likely reflective of the ever increasing popularity of YC or more people interested in being entrepreneurs.

# ### We can also see the seasonal progression

# In[ ]:


sns.countplot(df.batch)


# ## What type of company does YC fund?

# First let's get a sense of the total number of industries/fields that YC is invested in

# In[ ]:


print ("The total number of areas YC invests in", len(df.vertical.unique()))


# In[ ]:


(df.vertical.unique())


# There's a category with data not available. Let's call that category 'others'.

# In[ ]:


df['vertical']=['others' if pd.isnull(x) else x for x in df['vertical']]


# ## How many companies in each field has YC invested in?

# In[ ]:


sns.countplot(df.vertical)
plt.title('Type of companies funded')
plt.ylabel('# of companies')


# First let's get a sense of the total number of industries/fields that YC is invested in

# In[ ]:


print ("B2B companies form" ,round((df['vertical']=='B2B').value_counts()[1]/float(len(df))*100),"% of YC portfolio")


# We can also get a sense of investment in each field has progressed

# In[ ]:


sns.countplot(df['vertical'],hue=df['year'])


# In[ ]:





# In[ ]:




