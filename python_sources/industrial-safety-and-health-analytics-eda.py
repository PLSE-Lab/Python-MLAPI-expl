#!/usr/bin/env python
# coding: utf-8

# ## Note: WORK IN PROGRESS

# ## Importing data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# In[ ]:


data = pd.read_csv('/kaggle/input/industrial-safety-and-health-analytics-database/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')


# ## Data Profile

# In[ ]:


data.profile_report()


# ## A quick look at the data and initial action plan.

# In[ ]:


data.head(10)


# In[ ]:


print('There are '+str(data.shape[0])+'rows and '+str(data.shape[1])+' column in the dataset')
print('')
print('')
print('Columns in the dataset:')
for column in data.columns:
    print (column)
    
print('')
print('')
print('Brief information about the dataset:')
print(data.info())


# Initial action plan:
# 1. **Unnamed: 0**
#     - To be removed
# 2. **data** 
#     - to be renamed to **Date**
#     - to be converted to date type
#     - Check for trends over time and possibly even see monthly or day of the week trend if any.
#     - Country and locality wise trends over time.
# 3. **Countries**
#     - Count of incidents by country and further by locality
# 4. **Local**
#     - See **countries**
# 5. **Industry Sector**
#     - Count of incidents by industry sector.
#     - Checking relation between industry sector and country/localoty.
# 6. **Accident Level**
#     - Count of incidents by accident level.
# 7. **Potential Accident Level**
#     - Count of incidents by potential accident level.
#     - Comparison of actual and potention accident level.
#     - Checking the description where major incidents were averted/potentially low accidents turned major.
# 8. **Genre**
#     - Rename this to **Gender**
#     - Count of incidents by gender.
# 9. **Employee or Third Party**
#     - Count of incidents by Employee or Third Party.
# 10. **Critical Risk**
#     - Count of incidents by critical risk.
# 11. **Description**
#     - see point 7

# Dropping and renaming columns:

# In[ ]:


data.drop(labels='Unnamed_0',axis=1,inplace=True)
data.rename(columns={"Genre": "Gender", "Data": "Date"}, inplace=True)
data['Date']= pd.to_datetime(data['Date']) 
data.head(1)


# In[ ]:


data.info()


# In[ ]:


f, axes = plt.subplots(1,1,figsize=(15,5))
sns.lineplot(data = data['Date'].value_counts())


# In[ ]:


f, axes = plt.subplots(3,1,figsize=(15,5),sharex=True,sharey=True)
ax=0
for country in data['Countries'].unique():
    sns.lineplot(data = data[data['Countries']==country]['Date'].value_counts(),ax = axes[ax])
#    ax[0].set_title(country)
    ax=ax+1


# In[ ]:


#g = sns.FacetGrid(data, row='Countries')#row="smoker"
#g.map(sns.lineplot(data = data,x='date',y=data['Countries'].value_counts()))


# WIP..
