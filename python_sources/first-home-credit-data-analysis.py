#!/usr/bin/env python
# coding: utf-8

# # First data exploration 
# This notebook should allow me to quickly look into the application dataset and to drill into interesting sections. This is an ongoing effort, so whenever I find something interesting I am adding a new section.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
application = pd.read_csv('../input/application_train.csv')
print("Overall total applicants: %d" % application.size)


# Let's see what columns are in the main training dataset. Wow there are 122 columns, see below:

# In[ ]:


print(application.dtypes)


# ## Total number of granted applications
# Let's see how many applications were positive:

# In[ ]:


print(application.groupby(['TARGET']).TARGET.count())


# ## Lets see the categories of education
# Let's see how the education categories look like:

# In[ ]:


print(application['NAME_EDUCATION_TYPE'].head(10))


# ## Flag columns? 
# What are those flag columns? How many of those are flagged or not? Check an example 'mobile flag':

# In[ ]:


print(application['FLAG_MOBIL'].head(10)) 


# In[ ]:


import matplotlib.pyplot as plt

ed = application.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
u_ed = application.NAME_EDUCATION_TYPE.unique()
plt.figure(figsize=(15, 3))
plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
#plt.bar(ages, counts['F'], bottom=counts['M'], color='pink', label='F')
plt.legend()
plt.xlabel('Education level')
plt.plot()


# ## Children distribution
# Let's see the children count per application

# In[ ]:


import matplotlib.pyplot as plt

ed = application.groupby('CNT_CHILDREN').CNT_CHILDREN.count()
u_ch = application.CNT_CHILDREN.unique()
plt.figure(figsize=(10, 6))
plt.bar(u_ch, ed, bottom=None, color='green', label='Children count')
plt.legend()
plt.xlabel('Children count')
plt.plot()


# In[ ]:


ed = application.groupby('NAME_CONTRACT_TYPE').NAME_CONTRACT_TYPE.count()
u_ct = application.NAME_CONTRACT_TYPE.unique()
plt.figure(figsize=(5, 3))
plt.bar(u_ct, ed, bottom=None, color='grey', label='Contract type count')
plt.legend()
plt.xlabel('Contract type count')
plt.plot()


# ## Granted applications per number of children
# Granted applications stacked by number of children. 

# In[ ]:


ed = application.groupby(['TARGET', 'CNT_CHILDREN'])['TARGET'].count().unstack('TARGET').fillna(0)
ed.plot(kind='bar', stacked=True)
print(ed)


# ## Income distribution and target value
# Lets create income buckets with 10K steps and plot the granted applications compared to the declined ones in each income bucket.

# In[ ]:


application['income_bins'] = pd.cut(application['AMT_INCOME_TOTAL'], range(0, 1000000, 10000))

ed = application.groupby(['TARGET', 'income_bins'])['TARGET'].count().unstack('TARGET').fillna(0)
ed.plot(kind='bar', stacked=True, figsize=(50, 15))
print(ed)

