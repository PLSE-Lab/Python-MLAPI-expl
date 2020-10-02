#!/usr/bin/env python
# coding: utf-8

# # Goals
# 
#  * Check if h1b demand is growing
#  * Analyze which companies use h1b the most
#  * Calculate how much they (top companies using h1b) pay as compared to rest

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## General data processing

# In[ ]:


data = pd.read_csv('../input/h1b_kaggle.csv')


# In[ ]:


data.dropna(inplace=True)


# There are some duplicate employer names. Let's do normalization.

# In[ ]:


import re
data.EMPLOYER_NAME = data.EMPLOYER_NAME.apply(lambda x: re.sub(' +',' ', x.strip()))
data.JOB_TITLE = data.JOB_TITLE.apply(lambda x: re.sub(' +',' ', x.strip()))
data.WORKSITE = data.WORKSITE.apply(lambda x: re.sub(' +',' ', x.strip()))
data.SOC_NAME = data.SOC_NAME.apply(lambda x: re.sub(' +',' ', x.strip()))


# In[ ]:


data.columns


# In[ ]:


data.head()


# ## Check if h1b demand is growing

# In[ ]:


yearly = data.groupby('YEAR')
yearly['CASE_STATUS'].count()
yearly['CASE_STATUS'].count().plot()


# So, the demand is clearly growing. It would be interesting to see how this line correlates with number of job postings.

# ## Analyze which companies use h1b the most

# In[ ]:


years = [2011, 2012, 2013, 2014, 2015, 2016]
for year in years:
    filtered = data[data['YEAR'] == year]
    print(year)
    print('---')
    print(filtered.groupby('EMPLOYER_NAME')['CASE_STATUS'].count().sort_values(ascending=False).head(10))
    print('\n')


# So it's a consistent trend. Every year, max number of petitions are filed by indian outsourcing companies.

# ## Calculate how much they pay as compared to rest
# 
# Let's try to find if top users of h1b pay less than others

# In[ ]:


top10 = data.groupby('EMPLOYER_NAME')['CASE_STATUS'].count().sort_values(ascending=False).head(10).reset_index()['EMPLOYER_NAME']
top10wages = data[data['EMPLOYER_NAME'].isin(top10)]['PREVAILING_WAGE'].mean()
restwages = data[~data['EMPLOYER_NAME'].isin(top10)]['PREVAILING_WAGE'].mean()
print(top10wages, restwages)


# Woah. That's a big difference. But we should check this with location data, because salaries are heavily dependent on job location.

# In[ ]:





# In[ ]:


top10employers = data[data['EMPLOYER_NAME'].isin(top10)]
rest = data[~data['EMPLOYER_NAME'].isin(top10)]


# In[ ]:


top10employers.groupby(['WORKSITE', 'JOB_TITLE'])['PREVAILING_WAGE'].mean()


# In[ ]:


rest.groupby(['EMPLOYER_NAME', 'JOB_TITLE'])['PREVAILING_WAGE'].mean()

