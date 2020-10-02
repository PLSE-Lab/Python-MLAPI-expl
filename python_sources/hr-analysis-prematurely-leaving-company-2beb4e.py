#!/usr/bin/env python
# coding: utf-8

# HR Data - Prematurely Leaving Company

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


hr = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


hr.info()


# In[ ]:


hr.describe()


# In[ ]:


hr.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(hr)


# In[ ]:


sns.countplot(x='time_spend_company', data=hr)


# In[ ]:


hr.columns


# In[ ]:


sns.heatmap(hr.corr())


# In[ ]:


hr.columns


# In[ ]:


sns.jointplot(y='left', x='time_spend_company', data=hr)


# In[ ]:


sns.heatmap(hr.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.countplot(x='left',data=hr)


# In[ ]:


sns.boxplot(x='left',y='satisfaction_level',data=hr,palette='winter')


# In[ ]:




