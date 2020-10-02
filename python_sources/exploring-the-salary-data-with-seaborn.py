#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read in the data
data = pd.read_csv('../input/Salaries.csv', index_col='Id', na_values=['Not Provided', 'Not provided'])


# In[ ]:


## Pre-processing

# Change NaNs in the EmployeeName and JobTitle columns back to 'Not Provided'
data.loc[:, ['EmployeeName', 'JobTitle']] = data.loc[:, ['EmployeeName', 'JobTitle']].apply(lambda x: x.fillna('Not provided'))

# Normalize EmployeeName and JobTitle
data.loc[:, 'EmployeeName'] = data.loc[:, 'EmployeeName'].apply(lambda x: x.lower())
data.loc[:, 'JobTitle'] = data.loc[:, 'JobTitle'].apply(lambda x: x.lower())


# ### Distribution of Salaries

# In[ ]:


bPay = data[pd.notnull(data.BasePay)]

sns.distplot(bPay.BasePay, kde=False)
plt.title('Distribution of Base Pay')


# Looking at the distribution of base pay, we can see that it is bimodal. I suspect that this is full-time vs part-time employees. We can investigate further. Here I split the base pay data into a series for full-timers and a series for part-timers. 

# In[ ]:


PT = bPay[bPay.Status == 'PT']['BasePay']
FT = bPay[bPay.Status == 'FT']['BasePay']

PT.name = 'Part Time'
FT.name = 'Full Time'

plt.figure(figsize=(8,6))
sns.distplot(PT, kde=False)
sns.distplot(FT, kde=False)
plt.title('Distribution of Full-Time and Part-Time Employee Base Pay')


# ### Police salaries vs firefighter salaries

# In[ ]:


print('Median firefighter salary: $' + str(np.nanmedian(bPay[bPay.JobTitle == 'firefighter']['BasePay'])))
print('Median police salary: $' + str(np.nanmedian(bPay[bPay.JobTitle == 'police officer']['BasePay'])))


# In[ ]:


polFir = bPay[np.logical_or(bPay.JobTitle == 'firefighter', bPay.JobTitle == 'police officer')]

g = sns.FacetGrid(polFir, col='Year')
g.map(sns.violinplot, 'JobTitle', 'BasePay')


# In[ ]:


sns.distplot(bPay[bPay.JobTitle == 'firefighter']['BasePay'], kde=False, color='red')
sns.distplot(bPay[bPay.JobTitle == 'police officer']['BasePay'], kde=False)
plt.title('Distribution of Police and Firefighter Base Pay')


# Firefighters in San Francisco have a higher median salary at $109783.87 than police officers at $92615.14. Firefighter salaries also appear to have less variance. 

# ### Median Salary by Year

# In[ ]:


byYear = bPay.groupby('Year').aggregate(np.median)
byYear['Year'] = byYear.index

sns.barplot(data=byYear, x='Year', y='BasePay')

