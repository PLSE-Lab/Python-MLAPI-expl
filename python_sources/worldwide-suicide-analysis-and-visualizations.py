#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # ****Worldwide_Suicide_Analysis_and_Visualizations****

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


suicide = pd.read_csv('../input/master.csv', usecols = [0,1,2,3,4,5,11], index_col=['country', 'year']).sort_index()
suicide.head(10)


# ## Total Suicides Every Year all over the world

# In[ ]:


suicide.groupby('year').suicides_no.sum()


# In[ ]:


# Year wise Plot
plt.figure(figsize=(15,10))
suicide.groupby('year').suicides_no.sum().plot(kind = 'bar')
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15, rotation=0)
plt.show()


# ### Total Suicide from 1985 - 2016

# In[ ]:


suicide.suicides_no.sum()


# ### Male and Female suicides sum

# In[ ]:


suicide.groupby('sex').suicides_no.sum()


# ### Yearly comparison of male and male suicides all over the world

# In[ ]:


suicide.groupby(['year','sex']).suicides_no.sum().head(10)


# In[ ]:


# Yearly plot of Male and Female Suicides
plt.figure(figsize=(15,10))
suicide.groupby(['year','sex']).suicides_no.sum().plot(kind = 'bar')

plt.show()


# ### Country wise suicides from 1985 - 2016

# In[ ]:


suicide.groupby(['country']).suicides_no.sum().head(10)


# In[ ]:


suicide.groupby(['country']).suicides_no.sum().tail(10)


# ### Total Suicides in United States

# In[ ]:


suicide.groupby(['country']).suicides_no.sum().loc['United States']


# ### Total Suicides in Russian Federation

# In[ ]:


suicide.groupby(['country']).suicides_no.sum().loc['Russian Federation']


# In[ ]:


# Sorted country wise suicide Plot
plt.figure(figsize=(15,10))
suicide.groupby(['country']).suicides_no.sum().sort_values().plot(kind = 'bar')


# ### Age wise sum of suicides in all Countries from 1985 - 2016

# In[ ]:


suicide.groupby(['country','age']).suicides_no.sum().head(10)


# ### Age wise suicides in United States
# 

# In[ ]:


suicide.groupby(['country','age']).sum().loc['United States']


# ### Age wise suicides of all Countries

# In[ ]:


suicide.groupby(['age']).sum().head(12)


# In[ ]:


# Worldwide suicides by age group
plt.figure(figsize=(10,8))
suicide.groupby(['age']).suicides_no.sum().sort_values().plot(kind = 'bar')
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=15, rotation=0)
plt.show()


# In[ ]:


# World wide male and female age group suicides 
plt.figure(figsize=(10,8))
suicide.groupby(['age', 'sex']).suicides_no.sum().sort_values().plot(kind = 'bar')

plt.show()


# In[ ]:


suicide.groupby(['country','sex']).suicides_no.sum().head(10)


# ### Country wise male and female suicide Plot

# In[ ]:


plt.figure(figsize=(50,30))
suicide.groupby(['country', 'sex']).suicides_no.sum().plot(kind = 'bar')


# ### Plotting only male or female suicides all over the world

# In[ ]:


suicide2 = pd.read_csv('../input/master.csv')
suicide2.head()


# In[ ]:


df = suicide2.pivot_table(values='suicides_no', index='country', columns='sex',aggfunc='sum')
df.head()


# In[ ]:


df.tail()


# In[ ]:


df['Total_f_m'] = df['female'] + df['male']
df.head()


# ### Total suicides country wise

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(50,30))
df.groupby(['country']).Total_f_m.sum().plot(kind = 'bar')
plt.show()


# ### Selecting Individual Country and Sex

# In[ ]:


df.loc['United States','female']


# ### Top 20 countries with high Female suicides
# 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(50,30))
df.groupby(['country']).female.sum().sort_values().tail(20).plot(kind = 'bar')
plt.xticks(fontsize=25, rotation=0)
plt.yticks(fontsize=15, rotation=0)
plt.show()


# ### Top 20 countries with high Male suicides

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(50,30))
df.groupby(['country']).male.sum().sort_values().tail(20).plot(kind = 'bar')
plt.xticks(fontsize=25, rotation=45)
plt.yticks(fontsize=25, rotation=0)
plt.show()


# In[ ]:


suicide.groupby(['sex','country']).suicides_no.sum()


# In[ ]:


suicide.groupby(['sex','country']).suicides_no.sum().loc[('female','United States')]


# In[ ]:


suicide.groupby(['sex','country']).suicides_no.sum().loc[('male','Japan')]


# #### To be continued .........

# ##### Thanks for reading and suggestions are welcome. I ll be improving this kernel using seaborn and will do some more data analysis using other columns.
