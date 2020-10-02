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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Assignment 0 
# This is an assignment notebook for DATAI's Data Science Udemy Course
# <br>https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
# <br>https://www.udemy.com/course/data-science-sfrdan-uzmanlga-veri-bilimi-2/

# In[ ]:


df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
df.info()


# We have 6 strings and 6 numerical data, all of the non-null which is great that makes 12 total columns having 27820 data

# In[ ]:


# correlation map
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# I believe we can use HDI for year and gdp_per_capita colums for plotting bc they are directly proportional(0.8 almost 1) to eachother. Also, could use population and suicide_no columns which has good correlation.

# In[ ]:


df.head()

It turns out hdi for year is always NaN so I would go for population and suicides_no for scatter
# In[ ]:


df.columns


# In[ ]:


df_suicide = (df.suicides_no / 1000)+ 2000

df_suicide.plot(kind='line', label='Suicide No', linewidth=2, alpha=.5, grid=True, linestyle=':')
df.year.plot(label='Year', linewidth=2, alpha=.5, grid=True, linestyle='-.')

plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot')
plt.show()


# I really don't know what is the purpose of doing plot on the above but I just tried to get someting that makes sense in suicides numbers over years :))

# In[ ]:


df.plot(kind='scatter', x='population', y='suicides_no', alpha=.5)
plt.xlabel('Population')
plt.ylabel('Suicide No\'s')
plt.title('Population - Suicide No Scatter Plot')
plt.show()


# Except having numbers of population as 0-1-2-3-4, I would say we could fit a line in this graph around 10 degrees or something close to that and if population increases, suicide no is most likely will increase

# In[ ]:


df['suicides/100k pop'].plot(kind='hist',bins=10, density=1)


# For this graph we could say that suicide rates for 100k population is in range of 0-25 for 35 percent of the data

# In[ ]:


myFilter = (df.year > 2010) & (df['suicides/100k pop'] > 100)
df[myFilter]


# ## Continuing from Assignment 1 and further

# In[ ]:


#user defined function
def square(x):
    return x**2
print(square(4))


# In[ ]:


#scope
x=5
y = 4
def scopeExample():
    x=1
    return x*y
print(x)
print(scopeExample())


# In[ ]:


# default args
def calculatePower(x, y=2):
    return x**y
print(calculatePower(2))
print(calculatePower(2,3))


# In[ ]:


def expArgs(*args):
    return [i for i in args]
print(expArgs(1,2,3,4,5))


# In[ ]:


square = lambda x: x**2
print(square(2))


# In[ ]:


n = 'someone'
iterableN = iter(n)
print(next(iterableN))
print(*iterableN)


# In[ ]:


df.head()


# In[ ]:


threshold = sum(df['gdp_per_capita ($)'])/len(df['gdp_per_capita ($)'])
df['average'] = ['higher than average' if i > threshold else 'lower than average' for i in df['gdp_per_capita ($)']]
df.head()


# ### Assignment 2

# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


print(df['country'].value_counts(dropna =False))


# In[ ]:


df.describe()


# In[ ]:


df.boxplot(column='HDI for year')
plt.show()


# In[ ]:


df_ = df.head(10)
df_


# In[ ]:


melted = pd.melt(frame=df_,id_vars = 'generation', value_vars='suicides_no')
melted


# In[ ]:


melted = melted.head(2)
melted.pivot(index = 'generation', columns = 'variable',values='value')


# In[ ]:


df1 = df.head(10)
df2 = df.tail(10)
conc_df = pd.concat([df1,df2], axis=0, ignore_index=True)
conc_df


# In[ ]:


df1 = conc_df.population
df2 = conc_df.suicides_no
concat_df = pd.concat([df1,df2],axis=1)
concat_df


# In[ ]:


df.dtypes


# In[ ]:


df_copy = df.copy()
df_copy['year'] = df['year'].astype('object')
df_copy.dtypes


# In[ ]:


df.info()


# In[ ]:


df['HDI for year'].value_counts(dropna=False)


# In[ ]:


df_copy['HDI for year'].dropna(inplace=True)


# In[ ]:


assert df_copy['HDI for year'].notnull().all()
# Will return nothing if we successfully dropped null values.


# In[ ]:


df_copy['HDI for year'].value_counts(dropna=False)


# ### Assignment 3

# In[ ]:


country = ['Norway', 'Sweden', 'Denmark', 'Finland', 'Netherlands']
capital = ['Oslo', 'Stockholm', 'Copenhagen', 'Helsinki', 'Amsterdam']
label = ['country', 'capital']
list_col= [country, capital]
zipped = dict(zip(label, list_col))
new_df = pd.DataFrame(zipped)
new_df


# In[ ]:


new_df['continent']= 'Europe'
new_df


# In[ ]:


new_df['hdi_ranking'] = [1,7,11,15,10]
new_df


# In[ ]:


df1 = df.loc[:,['suicides_no', 'suicides/100k pop']]
df1.plot()
plt.show()


# In[ ]:


df1.plot(subplots=True)
plt.show()


# In[ ]:


df1.plot(kind='scatter', x='suicides_no', y='suicides/100k pop')
plt.show()


# In[ ]:


df1.plot(kind="hist", y="suicides_no", bins=50, range=(0,250), normed = True)
plt.show()


# In[ ]:


fig,axes = plt.subplots(nrows=2,ncols=1)
df1.plot(kind='hist', y='suicides/100k pop', bins=50, range=(0,100), normed=True, ax=axes[0])
df1.plot(kind='hist', y='suicides/100k pop', bins=50, range=(0,100), normed=True, ax=axes[1], cumulative=True)
plt.savefig('graph.png')
plt


# In[ ]:


df.describe()


# In[ ]:


time_list =  ['1992-03-08']
print(type(time_list[0]))
dto = pd.to_datetime(time_list)
print(type(dto))


# In[ ]:


df2 = df.head()
dList = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
dto = pd.to_datetime(dList)
df2['date'] = dto

df2 = df2.set_index('date')
df2


# In[ ]:


print(df2.loc["1993-03-16"])
print(df2.loc["1992-01-10":'1993-03-15'])


# In[ ]:


df2.resample('A').mean() #'M' = month, 'A'=year


# In[ ]:


df2.resample('M').mean()


# In[ ]:


df2.resample('M').first().interpolate('linear')


# In[ ]:


df2.resample('M').mean().interpolate('linear')
# now we will interpolate with mean unlike the first one which just fill the gap between two null data


# ### Assignment 4

# In[ ]:


df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
df.head()


# In[ ]:


df.loc[1,['age']]


# In[ ]:


df[['sex', 'age']]


# In[ ]:


print(type(df['age']))
print(type(df[['age']]))


# In[ ]:


df.loc[1:10,'year':'age']


# In[ ]:


ff = df.suicides_no > 10000
sf = df.year > 2010
df[ff & sf]


# In[ ]:


df.year[df.suicides_no>11000]


# In[ ]:


def div(n): return n/2
df['gdp_per_capita ($)'].apply(div)


# In[ ]:


df['gdp_per_capita ($)'].apply(lambda n : n/2)


# In[ ]:


df['no_sense'] = df.year / df.suicides_no
df.head()


# In[ ]:


print(df.index.name)
df.index.name = 'index'
df.head()


# In[ ]:


df3 = df.copy()
df3.index = range(100,27920)
df3.head()


# In[ ]:


df3 = df3.set_index(['population', 'suicides_no'])
df3.head()


# In[ ]:


dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df4 = pd.DataFrame(dic)
df4


# In[ ]:


df4.pivot(index='treatment', columns='gender',values='response')


# In[ ]:


df5 = df4.set_index(['treatment','gender'])
df5


# In[ ]:


df5.unstack(level=0)


# In[ ]:


df5.unstack(level=1)


# In[ ]:


df5 = df5.swaplevel(0,1)
df5


# In[ ]:


# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df4,id_vars="treatment",value_vars=["age","response"])


# In[ ]:


df4.groupby('treatment').mean()


# In[ ]:


df4.groupby('treatment').age.max()


# In[ ]:


df4.groupby('treatment')[['age', 'response']].min()

