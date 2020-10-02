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


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
pd.options.display.max_columns = 200
print(os.listdir("../input"))
data = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1")
data.columns
data.dtypes                          
np.iinfo('uint16')


# In[ ]:


data.info


# In[ ]:


data.describe


# In[ ]:


data.shape


# In[ ]:


data.head


# In[ ]:


data.tail


# In[ ]:


#District Wise Crime
plt.figure(figsize=(16,8))
data['DISTRICT'].value_counts().plot.bar()
plt.title('BOSTON: District wise Crimes')
plt.ylabel('Number of Crimes')
plt.xlabel('District')
plt.show()


# In[ ]:


#year wise crime trend
plt.figure(figsize=(16,8))
data['YEAR'].value_counts().plot.bar()
plt.title('BOSTON: Crimes - Yearly trend')
plt.ylabel('Number of Crimes')
plt.xlabel('Year')
plt.show()


# In[ ]:


#hour wise crime trend
plt.figure(figsize=(16,8))
data['HOUR'].value_counts().plot.bar()
plt.title('BOSTON: Crimes - Hourly trend')
plt.ylabel('Number of Crimes')
plt.xlabel('Hour')
plt.show()


# In[ ]:


#Weekly crime trend
plt.figure(figsize=(16,8))
data['DAY_OF_WEEK'].value_counts().plot.bar()
plt.title('BOSTON: Crimes - Weekly trend')
plt.ylabel('Number of Crimes')
plt.xlabel('Week of Day')
plt.show()


# In[ ]:


import seaborn as sns


# In[ ]:


labels = data['YEAR'].astype('category').cat.categories.tolist()
counts = data['YEAR'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,  autopct='%1.1f%%',shadow=True) 
ax1.axis('equal')
plt.show()


# In[ ]:


labels = data['HOUR'].astype('category').cat.categories.tolist()
counts = data['HOUR'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,  autopct='%1.1f%%',shadow=True) 
ax1.axis('equal')
plt.show()


# In[ ]:


sns.pairplot(data,hue ='UCR_PART') 


# In[ ]:


sns.kdeplot(data['MONTH'], data['YEAR'] )

