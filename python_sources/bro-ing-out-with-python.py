#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Maybe one too many adult sodas before this, but let's see
# Bro Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as bro_plt
import pandas as pd
import colorsys
bro_plt.style.use('seaborn-talk')

# Create the bro df
bro_df = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv", sep=',')
HSV_tuples = [(x*1.0/10, 0.5, 0.5) for x in range(10)]


# In[ ]:


bro_df.head()


# In[ ]:


# quite a few columns here LOL
bro_df.columns.values


# In[ ]:


eduLevels = bro_df['SchoolDegree'].astype('category')
c = len(eduLevels.value_counts())
bro_df.hist(by='SchoolDegree', column = 'HoursLearning', figsize=(15,15))


# In[ ]:


bro_df.hist(by='SchoolDegree', column = 'Income', figsize=(15,15))


# In[ ]:


labels = bro_df.Gender.value_counts().index
y_pos = np.arange(len(bro_df.Gender.value_counts()))
bars = bro_plt.barh(y_pos,bro_df.Gender.value_counts())
bro_plt.legend(bars, labels)
bro_plt.title("Gender")
bro_plt.show()


# In[ ]:


bro_df.Gender.value_counts()


# In[ ]:


bro_df['AttendedBootcamp'].hist()


# In[ ]:


campers = bro_df[bro_df['AttendedBootcamp']==1.0]
nerds = bro_df[bro_df['AttendedBootcamp']==0.0]


# In[ ]:


campers.hist(column = 'Income', figsize=(15,15))


# In[ ]:


nerds.hist(column = 'Income', bins=4, figsize=(15,15))


# In[ ]:




