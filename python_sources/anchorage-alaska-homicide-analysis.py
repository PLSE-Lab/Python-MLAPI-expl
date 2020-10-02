#!/usr/bin/env python
# coding: utf-8

# It might be fun to look at homicide trends in my hometown. So here goes.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/database.csv', index_col='Record ID', low_memory=False)
anc = data[data['City'] == 'Anchorage']
del anc['City']
del anc['Agency Code']
del anc['State']


# All righty then, that certainly narrows things down to a bite size chunk. Now to make some simple plots and get a quick feel for my data.

# In[ ]:


_ = sns.countplot(x='Year', data=anc)
plt.xticks(rotation=90)
plt.title('Number of Homicides Annuallly in Anchorage, AK')
plt.xlabel('Year')
plt.ylabel('Number of Homicides')
plt.show()


# Looks like there might be some kind of a periodic cycle going on here. It's hard not to notice the fairly even wave-like trend in the chart. But I can't let myself get distracted! Moving right along...

# In[ ]:


sns.countplot(x='Month', data=anc, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
plt.xticks(rotation=90)
plt.title('Murders by Month')
plt.xlabel('Month')
plt.ylabel('Number of Murders')
plt.show()


# In[ ]:


sns.countplot(x='Crime Type', data=anc)
plt.title('Homicide by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.countplot(x='Crime Solved', data=anc)
plt.title('Crime Solved')
plt.xlabel('Solved?')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# I think I'll look at age and sex together.

# In[ ]:


sns.distplot(anc['Victim Age'], norm_hist=True, kde=True, bins=50)
plt.title('Normalized Homicides by Age')
plt.xlabel('Age')
plt.ylabel('Normalized Frequency')
plt.show()


# There are a surprising number of these deaths among kids. Are these homicides or negligent manslaughter?

# In[ ]:


anc_kids = anc[anc['Victim Age'] < 16]
sns.countplot(x='Crime Type', data=anc_kids)
plt.title('Crime Type for Victims Under 16 Years Old')
plt.ylabel('Count')
plt.xlabel('Crime Type')
plt.show()


# It looks like the majority of the negligent manslaughter cases happened among kids. Interesting.
# 
# Now I'm curious to see if there is a correlation between attributes of the victims and perpetrators. For instance, is age of the perpetrator and the victim correlated? I'll visualize it first and then create the simple statistics to give a more quantitative answer.

# In[ ]:


x = anc['Victim Age']
y = anc['Perpetrator Age']
plt.scatter(x, y)
plt.title('Victim Age vs. Perpetrator Age')
plt.xlabel('Victim Age')
plt.ylabel('Perpetrator Age')
plt.xticks(rotation=90)
plt.show()


# This plot seems to suggest that some of the perpetrators were 0 years old. Is that really what I am seeing here?

# In[ ]:


print(anc['Perpetrator Age'].min())


# Yup. Well, that's disappointing. Clearly there are some data issues here. I'm going to do this again but this time I am going to get rid of all rows where either the perpetrator or the victim are age 0. I know I'm throwing out some valid data here, probably, but I will operate under the assumption that the general trends will hold true for everything that is not an infanticide or SIDs-type death.

# In[ ]:


anc['Perpetrator Age'] = anc['Perpetrator Age'].astype(int)
anc_not_infant = anc[anc['Victim Age'] > 0]
anc_not_infant = anc_not_infant[anc_not_infant['Perpetrator Age'] > 0]

x = anc_not_infant['Victim Age']
y = anc_not_infant['Perpetrator Age']
plt.scatter(x, y, c=['b', 'r'], cmap=anc['Victim Age'], alpha=.5)
plt.title('Victim Age vs. Perpetrator Age')
plt.xlabel('Victim Age')
plt.ylabel('Perpetrator Age')
plt.xticks(rotation=90)
plt.show()

print(np.corrcoef(x,y))


# Well...not that high, although even weak correlations like this can still be valid for social science purposes. I differentiated sex on this one by color just for fun. I don't really see anything significant arising from that decision, however.
# 
# Well, time to call it a night. I'll be back at this dataset as soon as I have the time.
