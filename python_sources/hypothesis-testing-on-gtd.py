#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/gtd/globalterrorismdb_0718dist.csv", encoding="ISO-8859-1")
df.head()


# Frst we will show the top 5 most affected countries

# In[ ]:


print(df['country_txt'].value_counts().head())
df['country_txt'].value_counts().head().plot(kind = 'bar')
plt.title("Terrorist attacks for top 5 most affected counntries")
plt.xlabel("Country")
plt.ylabel("Value_Counts")
plt.xticks(rotation = 80)


# * Our null hypothesis is  - Two contries were equally affected by the terrorism activities. 
# On what parameters?
# * Average number of kills on all attacks per year
# * We use two sided two sample t-test for equality of means under the assumption that the number of kills per attack binned per year is normally distrbuted
# * H0 : u1 = u2
# * H1 : u1 > u2

# In[ ]:


dict(df['country_txt'].value_counts())


# In[ ]:


df.set_index(df['country_txt'])


# We will consider Cameroon and Honduras for our analysis

# In[ ]:


df2 = df[(df['country_txt'] == 'Cameroon') | (df['country_txt'] == 'Honduras')]
df2.head()


# We generate the data for number kills per year of the two countries under consideration

# In[ ]:


d1 = dict(df2[df2['country_txt'] == 'Honduras'].groupby("iyear").mean()['nkill'].dropna())
d2 = dict(df2[df2['country_txt'] == 'Cameroon'].groupby("iyear").mean()['nkill'].dropna())


# In[ ]:


d1


# In[ ]:


d2


# Now our test procedure would be as follows https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm 

# In[ ]:


#Calculating Average number kills per year
Y1 = sum(d1.values())/len(d1)
Y2 = sum(d2.values())/len(d2)


# In[ ]:


#Calculating sample variance for the given data
s1_sqr = 0
for i in d1.values():
    s1_sqr += (i - Y1)**2
s1_sqr/=(len(d1) - 1)

s2_sqr = 0
for i in d2.values():
    s2_sqr +=(i - Y2)**2
s2_sqr/=(len(d2)-1)
s2_sqr


# In[ ]:


#Calculating test Statistic
T = (Y1 - Y2)/(s1_sqr/len(d1) + s2_sqr/len(d2))**0.5


# In[ ]:


# level of significance alpha = 0.05
alpha = 0.05
degree_of_freedom = len(d1) + len(d2) - 2


# Importing required package for finding the critical value of t distribution

# In[ ]:


from scipy import stats


# In[ ]:


t = stats.t.ppf(1-alpha,degree_of_freedom)


# In[ ]:


if abs(T) > t:
    print("Null Hypothesis is regected")
else:
    print("fail to regect Null Hypothesis ")


# We note that our null hypothesis is regected in favour of alternate hypothesis

# Which shows that Honduras is affected more than Camroon in terms of terrorism related killings per year

# In[ ]:




