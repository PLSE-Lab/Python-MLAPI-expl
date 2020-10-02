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


import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()


# In[ ]:


print(df.shape) # show's there are 215 samples and 15 features
df.describe() # Salary has count 148, thus it has NaN Values, which was because they are not placed, so their salary can be assigned to zero


# In[ ]:


df.corr()


# # (i) Histogram for Degree_P
# ## Fig1
# Use appropriate bin width 

# In[ ]:


#  Draw a histogram for degreep.
degree_p_vals = df["degree_p"]
w = 2
n = int((degree_p_vals.max() - degree_p_vals.min())/w)
degree_p_vals.hist(bins=n)


# ## Differentiate based on status (Fig 2)

# In[ ]:



degree_p_vals_sp = df[df["status"]=="Placed"]["degree_p"]
degree_p_vals_snp = df[df["status"]!="Placed"]["degree_p"]
degree_p_vals_sp.hist(bins=n,color='lightgreen',alpha=0.7,label="Placed")
degree_p_vals_snp.hist(bins=n,color='firebrick',alpha=0.5,label="Not Placed")
plt.legend()


# ## Using faceting introduce gender 
# ## Fig 3

# In[ ]:


selector = 'gender'
selector_val="M"
degree_p_vals_sp = df[df[selector]==selector_val]["degree_p"]
degree_p_vals_snp = df[df[selector]!=selector_val]["degree_p"]
degree_p_vals_sp.hist(bins=n,color='lightgreen',alpha=0.7,label="Male")
degree_p_vals_snp.hist(bins=n,color='firebrick',alpha=0.5,label="Female")
plt.legend()


# # (ii) Draw a boxplot for degreep 
# ## Fig4.

# In[ ]:


df["degree_p"].plot(kind='box')


# ## Fig 5

# In[ ]:


selector = 'gender'
selector_val="M"
degree_p_vals_M = df[df[selector]==selector_val]["degree_p"]
degree_p_vals_FM = df[df[selector]!=selector_val]["degree_p"]

fig, axs = plt.subplots(1,2)
fig.suptitle('Box Plot for Degree % based on gender')

degree_p_vals_M.plot(kind='box',ax=axs[0],title='Male')
degree_p_vals_FM.plot(kind='box',ax=axs[1],title='Female')
plt.legend()


# ## Using faceting introduce Using faceting introduce workex and gender together  and gender together 
# ## Fig 6

# In[ ]:


from sklearn.preprocessing import LabelEncoder
df_f = df[["workex","gender","degree_p"]].copy()
fig, axs = plt.subplots(1,4)
index = 0
for title,data in df_f.groupby(['workex','gender']):
    title = list(title)
    if title[1]=="F":
        title[1] = "Female"
    else:
        title[1] = "Male"
    
    if 'Y' in title[0]:
        title[0] = ' \nwith \nwork exp'
    else:
        title[0] = ' \nwithout \nwork exp'
    
    data.plot(kind='box',ax=axs[index],title=title[1]+title[0])
    index = index + 1


# # (iii) Draw a density plot for degree_p
# ## Fig 7

# In[ ]:


import seaborn as sns
sns.distplot(df['degree_p'])


# ## Fig 8

# In[ ]:


# Differentiate based on status 
import seaborn as sns
fig, axs = plt.subplots(1,2)
index = 0
for title,data in df.groupby('status'):
    print(title)
    sns.distplot(data['degree_p'],ax=axs[index],axlabel=title)
    index = index+1
plt.legend()


# ## Fig 9

# In[ ]:


# Differentiate based on status 
import seaborn as sns
fig, axs = plt.subplots(1,2)
index = 0
for title,data in df.groupby('status'):
    print(title)
    sns.distplot(data['degree_p'], hist=True, kde=True, 
                 bins=n, ax=axs[index],
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'color': 'blue'})
    index = index+1


# # (iv) Which of the above three types of plots is more useful for interpretation and how
# Ans)
# I have notices below observation from above plots
# * Fig 1 show we can observe that histogram is left scewed.
#     * Majority of them lie in in 62% to 72%.
# * From Fig 2 it can be observed that 
#     * person with degree percentage above 80 % has high chances of getting placed.
#     * person with degree percentage below 55 % has lowest chances of getting placed.
# * With reference to Fig 2 and Fig 3 it can be said that out of 10 min 8 would be Female with degree percente above 80%
# * From Fig 4, we can identify there are few outlier and major of them are from Female
# * Fig 6, implies the Percentage seems most percentage is around 65 %, except for Female with work experience, they tend to have higher percentage which would be around 70%
# 
# ## I would prefer to use Histogram as it had helped in analysing most of the information, but as a confirmation other plots helps in confirmation and deeper understanding of the data

# In[ ]:




