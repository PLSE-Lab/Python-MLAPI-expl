#!/usr/bin/env python
# coding: utf-8

# # Just playing around with some healthcare data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/states.csv', header=0)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns


# ## Convert percentages from object to float.

# In[ ]:


str_percent_to_float = lambda x: float(x.strip(' ').strip('%'))/100
percent_to_convert = ['Uninsured Rate (2010)', 'Uninsured Rate (2015)', 'Uninsured Rate Change (2010-2015)']
df[percent_to_convert] = df[percent_to_convert].applymap(str_percent_to_float)


# ## Convert dollars from object to int.

# In[ ]:


str_dol_to_int = lambda x: int(x.strip('$'))
df['Average Monthly Tax Credit (2016)'] = df['Average Monthly Tax Credit (2016)'].map(str_dol_to_int)


# ## Convert bool to int.

# In[ ]:


df.drop([51], inplace=True) # getting rid of total US info ... (maybe stupid)
df['State Medicaid Expansion (2016)'] = df['State Medicaid Expansion (2016)'].map(lambda x: int(x))


# In[ ]:


df_without_state = df.drop(['State'], axis=1)


# In[ ]:


df_without_state.info()


# In[ ]:


# Check how many states saw a decrease in uninsured...
from collections import Counter
uninsured_rate_change = df['Uninsured Rate Change (2010-2015)']
Counter([np.sign(change) for change in uninsured_rate_change[:-1]]).most_common()


# ## Every state has insured more people since 2010

# In[ ]:


sns.barplot(x=df.State[:-1], y=uninsured_rate_change[:-1])
plt.xticks(rotation='vertical')
plt.xlabel('State')
plt.ylabel('Mean uninsured rate change (2010-2015)')


# ## Who has seen the largest decrease in uninsured?

# In[ ]:


# sort the values
largest_decrease = uninsured_rate_change.sort_values()

# plot the top 15 (ie, the states who have seen most decrease in uninsured)
sns.barplot(x=df.State.ix[largest_decrease[:15].index], y=largest_decrease[:15])
plt.xticks(rotation=45)
plt.xlabel('State')
plt.ylabel('Mean uninsured rate change (2010-2015)')

