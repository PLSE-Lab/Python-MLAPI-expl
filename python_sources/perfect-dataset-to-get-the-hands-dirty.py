#!/usr/bin/env python
# coding: utf-8

# ***US Jobs - Monster Analysis* **
# 
# which organization has many jobs...how frequent they are posting the jobs...
# which location has many jobs
# which organization has highes, lowest and avergae ssalary
# which job or role is more/less popular and what its salary
# highest full time employee salary??
# top 10 jobs by industry, salary, job type, location,

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df=pd.read_csv("../input/monster_com-job_sample.csv")


# In[ ]:


df.head(100)


# In[ ]:


df.info()


# **There are many columns which have null values such as date_added, job_type, organization, salary, and sector. We cannot predict anything in Machine Learning with the dirty data. Before applying any Machine Learning technique, we need to deal with the missing data and outliers. **

# **There are some columns which are useless and very obvious. For example, country, country code, has_expired, 
# and job_board. We are analysing this problem for US and Monster.com. And, Has_expired has only one value i.e. NO. So there is no point to use these columns further.**

# In[ ]:


df=df.drop(['country','country_code','job_board','has_expired'], axis=1)
df.head()


# **Also, there are some columns which are not required for analysis such as uniq_id and page_url. We can simpley drop these many columns.
# **

# In[ ]:


df=df.drop(['page_url','uniq_id'],axis=1)
df.head()


# In[ ]:


df.describe()


# ![](http://)**After watching carefully at the summary, this dataset is horrible. Out of 22000 there are only 122 values for date_added.
# In some rows of the Location column it has job_description and also salary column is not standardized and has many
# null values.**

# **The Location column is also messed up. It has many unused information like job descritpion, contact person, job title etc. To deal with this - generally the location field has city name, state, and zip code, so, I assume the location field won't be having characters more than 15-20 characters. 
# **

# In[ ]:


location=df['location'].str.split(',')
df['location']=location.str[0]
df=df[df['location'].apply(lambda x: len(x)<20)]
df.head()


# **In Job_type column, there are several types of field g iven such as 'Full Time', 'Full Time Employee', 'Part Time Employee',  'Contract', 'Project ', 'Temporary', 'Intern'. Again, the data is not properly formatted. **

# In[ ]:


jobtype=df['job_type'].str.split(',')
df['job_type']=jobtype.str[0]
df.head(100)


# **Since, 'Full Time Employee' and 'Full Time' positions are same. We need to combine these two into one.**

# In[ ]:





# In[ ]:





# In[ ]:


a


# In[ ]:


df.describe()


# In[ ]:




