#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:



d = pd.read_csv('../input/ttl-fb/TTL_PAGE.csv')


# In[ ]:


d.head()


# In[ ]:


d.columns


# In[ ]:


d.drop(d.index[1], inplace=True)


# In[ ]:


d


# In[ ]:


d.iloc[0:,5]


# In[ ]:


d = d.rename(columns={'Daily Page Engaged Users':'Daily_Engaged_Users','28 Days Page Engaged Users':'Monthly_Engaged_Users','28 Days Total Reach':'Total_Monthly_Reach'})


# In[ ]:


d


# In[ ]:


d.Daily_Engaged_Users


# In[ ]:


d.tail()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(30,20))
plt.title("Daily organic reach over time")
sns.scatterplot(x = d['Date'],y = d['Daily Organic Reach'])


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.title("Daily organic reach over a month")
sns.barplot(x = d.index,y = d['Daily Organic Reach'])
plt.xlabel("Number of days")
plt.ylabel("Organic reach")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.title("Total Monthly Reach")
sns.barplot(x = d.index,y = d['Total_Monthly_Reach'])
plt.xlabel("Number of days")
plt.ylabel("Monthly reach")


# In[ ]:


l = max(d.Monthly_Engaged_Users)
l


# In[ ]:


d.Monthly_Engaged_Users.mode()


# In[ ]:


d.describe()


# In[ ]:


d.Monthly_Engaged_Users.unique()


# In[ ]:


d.Monthly_Engaged_Users.value_counts


# In[ ]:


k = d[['Date', 'Daily_Engaged_Users', 'Monthly_Engaged_Users','Weekly Page Engaged Users','Lifetime Total Likes','Daily Total Reach','Weekly Total Reach','Total_Monthly_Reach']]
k


# In[ ]:


k.to_csv("social_media.csv")


# In[ ]:


k


# In[ ]:


k.head()


# In[ ]:


page = pd.read_csv("social_media.csv",index_col="Date", parse_dates=True)


# In[ ]:


page.head()


# In[ ]:


page


# In[ ]:


page_data = pd.read_csv("./social_media.csv")


# In[ ]:


page_data


# In[ ]:


page_data.head()


# In[ ]:


page_data.iloc[0:,0].dtype


# In[ ]:


page_data.Monthly_Engaged_Users.dtype


# In[ ]:


page_data.Daily_Engaged_Users.dtype


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.title("Daily visitors in the page")
sns.barplot(x = page_data.index,y = page_data['Daily_Engaged_Users'])
plt.xlabel("Number of days")
plt.ylabel("Daily visitors in the page")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(14,7))
plt.title("Daily visitors in the page")
sns.scatterplot(x = page_data.index,y = page_data['Daily_Engaged_Users'])
plt.xlabel("Number of days")
plt.ylabel("Daily visitors in the page")


# In[ ]:


page_data.columns


# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Daily visitors in the page")
sns.swarmplot(x = page_data.index,y = page_data['Daily_Engaged_Users'])
plt.xlabel("Number of days")
plt.ylabel("Daily visitors in the page")


# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Monthly visitors in the page")
sns.barplot(x = page_data.index,y = page_data['Monthly_Engaged_Users'])
plt.xlabel("Number of days")
plt.ylabel("Monthly visitors in the page")


# In[ ]:


plt.figure(figsize=(14,7))
plt.title("Monthly visitors in the page")
sns.scatterplot(x = page_data.index,y = page_data['Monthly_Engaged_Users'])
plt.xlabel("Number of days")
plt.ylabel("Monthly visitors in the page")


# In[ ]:


pd.Series([1, 2], dtype='int32')


# In[ ]:


page_data.Monthly_Engaged_Users.dtype
page_data.head()


# In[ ]:


feature = ['Date', 'Daily_Engaged_Users', 'Monthly_Engaged_Users',
       'Weekly Page Engaged Users', 'Lifetime Total Likes',
       'Daily Total Reach', 'Weekly Total Reach']


# In[ ]:


X = page_data[feature]


# In[ ]:


X.describe()


# In[ ]:


X.head()


# In[ ]:


y = page_data.Total_Monthly_Reach


# In[ ]:


y.head()


# In[ ]:


y


# In[ ]:




