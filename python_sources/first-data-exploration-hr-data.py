#!/usr/bin/env python
# coding: utf-8

# #This notebook is a simple data exploration of the core data set. 
# 
# I'll first try to figure out what the data looks like
# 
# Second will be trying to find correlations and to clear out empty fields 
# 
# third will be trying to find correlations in the data and to show what HR data can provide in terms of insights in the organisational population.
# 
# Also, this is my first kaggle project so I have to acknowledge a lot of people here on kaggle. First of all, the people who told me about kaggle, this is really eye opening. Second, the people who published their kernels so people could learn from their code
# 

# In[39]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# The data file location has changed, it's now in the folder called 'human-resources-data-set'.

# In[40]:


df =  pd.read_csv('../input/human-resources-data-set/core_dataset.csv')


# Below a first exploration of the data. Starting with the first few rows. Next some details of the data itself.

# In[41]:


df.head()


# In[42]:


df.info()


# In[43]:


df.columns


# In[44]:


df.select_dtypes(include=['float64']).columns.values


# Now, lets check the missing values in the data set

# In[45]:


df_isnull = (df.isnull().sum() / len(df))*100
df_isnull = df_isnull.drop(df_isnull[df_isnull ==0]).sort_values(ascending = False)[:30]
missing_data = pd.DataFrame({'Missing Ration' :df_isnull})
missing_data


# Date of termination obviously isn't always filled in. Lets fill that up with 'none', we could always change the 'None' values with some numerical stuff.

# In[46]:


df['Date of Termination'] = df['Date of Termination'].fillna("None")


# In[47]:


df_isnull = (df.isnull().sum() / len(df))*100
df_isnull = df_isnull.drop(df_isnull[df_isnull ==0].index).sort_values(ascending = False)
missing_data = pd.DataFrame({'Missing Ration' :df_isnull})
missing_data


# In[48]:


df.tail()


# I get the feeling that some lines are simply not filled, perhaps an error in the data set. Looking at the df.tail() is see on line 301 empty fields. Lets assume this is the case in the rest of the data set as well and clear those rows, we simply don't need those.

# In[49]:


df = df[df.Position.notnull()]


# In[50]:


df_isnull = (df.isnull().sum() / len(df))*100
df_isnull = df_isnull.drop(df_isnull[df_isnull ==0].index).sort_values(ascending = False)
missing_data = pd.DataFrame({'Missing Ration' :df_isnull})
missing_data


# Great! No more missing values!

# It's time to visualize some of the data starting with a correlation plot to see our first entry of future exploring.

# In[51]:


corrmat = df.corr()
plt.subplots(figsize=(4,4))
sns.heatmap(corrmat, vmax=0.9, square=True)


# Pay Rate seems to be related with Zip code. Lets explore that in more detail!

# In[52]:


sns.distplot(df['Pay Rate'])


# In[53]:


print(df['Pay Rate'].describe())
print("\nMedian of pay rate is: ", df['Pay Rate'].median(axis = 0))


# In[54]:


sns.regplot( x = 'Zip', y = 'Pay Rate', data = df)


# In[55]:


plt.figure(figsize = (16, 10))

sns.boxplot(x = 'Zip', y = 'Pay Rate', data = df)


# I'm not convinced that Zip code and Pay Rate is really correlated. It makes sense though that matplotlib sees a correlation since most of the employees live in a certain Zip code area.

# Let's explore that relation between age and Pay Rate

# In[56]:


sns.regplot( x = 'Age', y = 'Pay Rate', data = df)


# I realize now that it is quite inconvenient that there are spaces in the column names. So, I am going to do something about that.

# In[57]:


df.rename(columns={
    'Pay Rate': 'PayRate',
    'Employee Name': 'EmployeeName',
    'Employee Number': 'EmployeeNumber',
    'Hispanic/Latino': 'HispLat',
    'Date of Hire': 'DateHire',
    'Days Employed': 'DaysEmployed',
    'Date of Termination': 'DateTerm',
    'Reason For Term': 'ReasonTerm',
    'Employment Status': 'EmployStatus',
    'Manager Name': 'ManagerName',
    'Employee Source': 'EmployeeSource',
    'Performance Score': 'PerformanceScore'

}, inplace=True)


# In[58]:


df.head()


# Looks better now!

# I want to change hispanic latino into numerical data by saying No becomes a zero and a Yes becomes a one.

# In[59]:


HispLat_map ={'No': 0, 'Yes': 1, 'no': 0, 'yes': 1}
df['HispLat'] = df['HispLat'].replace(HispLat_map)
df['HispLat']


# And probably good to do the same for Sex but first explore how sexes are written down (Capitals or not).

# In[60]:


pd.crosstab(df.CitizenDesc, df.Sex)


# Ok so in this case, females become a zero and males a one.

# In[61]:


Sex_map ={'Female': 0, 'Male': 1, 'male': 0}
df['Sex'] = df['Sex'].replace(Sex_map)
pd.crosstab(df.CitizenDesc, df.Sex)


# Doing a little bit of Sexes exploration...

# In[62]:


pd.crosstab(df.State, df.Sex)


# Trying to figure out whether HispLat and PayRate are related.

# In[63]:


sns.violinplot('HispLat', 'PayRate', data = df)


# Women and man and PayRate...

# In[64]:


sns.violinplot('Sex', 'PayRate', data = df)


# And what about the performance scores?

# In[65]:


pd.crosstab(df.Sex.values, df.PerformanceScore.values)


# Interesting. This will probably need some exploration. Not yet though, first I will explore other stuff first such as marital status, HispLat and Performance over the sexes and the PayRate

# In[66]:


g = sns.FacetGrid(df, col='Sex', row='MaritalDesc')
g.map(plt.hist, 'PayRate')


# In[67]:


g = sns.FacetGrid(df, col='HispLat', row='MaritalDesc')
g.map(plt.hist, 'PayRate')


# In[68]:


g = sns.FacetGrid(df, col='Sex', row='PerformanceScore')
g.map(plt.hist, 'PayRate')


# Interesting stuff!
# 
# Looking forward to explore the performance score even more. Lets start this by checking the mean of PayRate and Age in combination with this score.

# In[69]:


df[['PerformanceScore', 'PayRate', 'Age']].groupby(['PerformanceScore'], 
as_index=False).mean()


# Let's group the PerformanceScores like this:
# 
# 90-day meets = 2
# Exceeds = 3
# Exceptional = 4
# Fully Meets = 2
# N/A = 0
# Needs Improvement = 1
# PIP = 1
# 

# In[70]:


PerfScore_map = {'90-day meets': 2, 'Exceeds': 3, 'Exceptional': 4, 'Fully Meets': 2, 'N/A- too early to review': 0,
                'Needs Improvement': 1, 'PIP': 1}

df['PerformanceScore'] = df['PerformanceScore'].replace(PerfScore_map)


# In[71]:


df.head()


# Ok how does working for a specific manager impact your performance score?

# In[72]:


g = sns.FacetGrid(df, col='Sex', row='ManagerName')
g.map(plt.hist, 'PerformanceScore')


# Working for Janet King as a woman gives you a big change of being graded with a higher score that 'Fully Meets' (mapped as 2). Working for Michael Albert though, is a little bit of since you'll have more change of needing improvement if you're a woman.

# Exploring where your employees come from and their performance might give you good insight in where you should find your next employee

# In[73]:


pd.crosstab(df.EmployeeSource.values, df.PerformanceScore.values) 


# In[74]:


g = sns.FacetGrid(df, row='EmployeeSource')
g.map(plt.hist, 'PerformanceScore')


# Apparently, looking at the above plot, employees receive a higher performance score when they are coming from the professional society and through MBTA ads.
# 
# Word of mouth is not so well for the performance score...

# In[75]:


sns.boxplot(y = 'EmployeeSource', x = 'PerformanceScore', data = df)


# A nice boxplot shows it even better ^

# Now let's see which manager gives you as an employee the change for a high performance score:

# In[77]:


sns.boxplot(y = 'ManagerName', x = 'PerformanceScore', data = df)


# Working for a manager drastically impacts your performance score. Perhaps good to explore this further with the other data sources. Maybe we can find out why?

# In[78]:


sns.boxplot(y = 'PayRate', x = 'PerformanceScore', data = df)


# According to above plot, you see the median being exceptionally high compared with the others for the pay rate and scoring a exceptional performance.

# Let's find out if race description has impact on performance and pay rate:

# In[79]:


sns.boxplot(y = 'RaceDesc', x = 'PerformanceScore', data = df)


# In[80]:


sns.boxplot(y = 'RaceDesc', x = 'PayRate', data = df)


# And your Marital description:

# In[81]:


sns.boxplot(y = 'MaritalDesc', x = 'PerformanceScore', data = df)


# In[82]:


sns.boxplot(y = 'MaritalDesc', x = 'PayRate', data = df)


# The performance and the reason of terminating the contract between organisation and employee

# In[83]:


sns.boxplot(y = 'ReasonTerm', x = 'PerformanceScore', data = df)


# Performance and performance score is a key reason for breaking up. Women who went on maternity leave and didn't return actually performed very well. Too bad they left the organisation!
# 
# Overal you can clearly see that the break up and the performance are related.

# What else is there to find in this core data set?
