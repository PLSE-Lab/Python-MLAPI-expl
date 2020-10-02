#!/usr/bin/env python
# coding: utf-8

# *Hey Friends! So this is my very first notebook. We all need to start from somewhere, so here is my starting.
# So we are here, cleaning and learning from the udemy data. Follow along the notebook for more info :) Thanks*

# In[ ]:


# few basic libraries to always import, to make your data cleaning and visualization easy and understandable

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# for getting the file location
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        name = os.path.join(dirname, filename)


# In[ ]:


# using pandas to read our csv file
df = pd.read_csv(name) 

# used to display a chunk of data
df.head()


# In[ ]:


# describe() is a very powerful function. It is used to summarise our numerical columns. 
# it provides the mean, standard deviation, count, etc
df.describe()


# In[ ]:


# info() functions shows the column details and the data type of the columns
df.info()


# In[ ]:


print(df.course_id.count() , df.course_id.nunique() , df.course_title.nunique() )


# In[ ]:


df.course_title.nunique()


# In[ ]:


print(df['url'].value_counts() > 1)


# In[ ]:


df['subject'].value_counts()


# In[ ]:


df.groupby(['subject'])['num_lectures'].agg('sum')


# In[ ]:


df.groupby(['subject'])['num_reviews'].agg('sum') # num_subscribers


# In[ ]:


df.groupby(['subject'])['num_subscribers'].agg('sum') # num_subscribers


# In[ ]:


df.groupby(['subject'])['price'].agg('mean') # num_subscribers


# In[ ]:


df.groupby(['subject'])['price'].agg('median') # num_subscribers


# In[ ]:


df[df['subject'] == 'Web Development'].describe()

