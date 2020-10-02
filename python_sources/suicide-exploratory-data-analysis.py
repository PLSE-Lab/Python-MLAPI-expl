#!/usr/bin/env python
# coding: utf-8

# In this kernel, I tried to showcase the effects of most of the independent variables on dependent variable(suicide_rate).
# 
# 1. Countrywise average suicide rate
# 2. Yearly suicide rate 
# 3. Country + gender suicide rate
# 4. Sucide rate by gender
# 5. Age + gender suicide rate 
# 6. Generation Analysis
# 7. Generation +  Age suicide rate
# 8. Generation + Gender suicide rate analysis
# 9. GDP per capita for country in every year.

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

import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[ ]:


data = pd.read_csv('../input/master.csv')


# In[ ]:


data.head()


# ## Checking the data

# In[ ]:


data.isna().sum()


# In[ ]:


data['country'].unique()


# In[ ]:


len(data['country'].unique())


# There are some 101 countries

# In[ ]:


data['country'].value_counts()


# Let's check if country-year and year in our data is same? if country-year contains country name then year then it's not useful to use redundant information.

# In[ ]:


data['country_year'] = data['country-year'].apply(lambda x: int(re.findall("[0-9]+",x)[0]))


# In[ ]:


data['country_year']


# In[ ]:


## checking if all years and country-year are equal
set(data['year'] == data['country_year'])


# Hence, from above code we can conclude that it's same. Let's drop country_year as no need .

# In[ ]:


data = data.drop(['country_year'], axis=1)


# In[ ]:


data.columns


# >> Removing "years"(string) from values.**

# In[ ]:


data['age'] = data['age'].replace(regex = {"years":''})


# In[ ]:


data['age']


# Different age-groups are these -:

# In[ ]:


data['age'].unique()


# In[ ]:


data['sex'].unique()


# In[ ]:


data['generation'].unique()


# In[ ]:


data['generation'].value_counts()


# In[ ]:


data[(data['country'] == 'Albania') & (data['year'] == 1987)].sort_values(by='age')


# ## Average suicide rate for each country.

# In[ ]:


country_suicide = data.groupby('country').agg('mean')['suicides_no'].sort_values(ascending=False)


# In[ ]:


country_suicide.values


# As we can see from above output that Russian Federation, USA, Japan has highest number of suicide rates of more than 2k and Russian Federation 

# In[ ]:


x = list(country_suicide.keys())
y = list(country_suicide.values)
plt.figure(figsize=(12,16))
plt.barh(x,y)
plt.show


# >> Russian Federation, United States, Japan has high number of suicides rate.

# ## Yearly suicides rate analysis

# In[ ]:


plt.figure(figsize = (8,6))
x = list(data.groupby(['year']).agg('sum')['suicides_no'].keys())
y = list(data.groupby(['year']).agg('sum')['suicides_no'].values)
plt.title("Yearly suicides rate")
plt.xlabel("Years")
plt.ylabel("suicide no.")
plt.bar(x, y)
#plt.set_xticklabels(tic)
plt.show()


# >> In 2016 it is very less.
# 
# >> After 1990 suicides rate will start increasing.

# In[ ]:


sns.lmplot(x="year", y="suicides_no", data=data)


# ## Country+year suicide rate analysis

# In[ ]:


data.groupby('country').agg(['min','max'])['suicides_no'].sort_values(by='max', ascending=False)


# In[ ]:


fig, ax = plt.subplots(figsize=(16,16))
country_year_average = data.groupby(['year', 'country']).agg('mean')['suicides_no']
country_year_average.unstack().plot(ax=ax)
country_year_average  = country_year_average.reset_index()


# ## Plotting only countries having suicides number greater than 1000 with year
# 

# In[ ]:


country_year_average = country_year_average[country_year_average['suicides_no']>1000]


# In[ ]:


country_year_average


# In[ ]:


country_year_average.sort_values(by='suicides_no', ascending=False)


# In[ ]:


country_year_average['year']


# In[ ]:


# fig, ax = plt.subplots(figsize=(16,16))
# data.groupby(['country', 'year']).agg('mean')['suicides_no'].unstack().plot(kind='violin',ax=ax)

#country_year_average = data.groupby(['country', 'year']).agg('mean')['suicides_no']
#plt.subplots(figsize=(8,8), )
plt.figure(figsize=(8,8))
g = sns.FacetGrid(country_year_average, col='year', height= 6, size=4, col_wrap=5)
g.map(plt.bar, "country", "suicides_no")
g.set_xticklabels(rotation=30,fontsize=10)


# In the above diagrams, we have plotted countries having suicides rates > 1000 are considered only so, we have Japan, Republic of Korea, Russian Federation, United States. In some years some of the countries doesn't have any suicides rate.

# ## Checking suicides_no by age-group
# 

# In[ ]:


plt.figure(figsize=(4,8))
x = list(data.groupby(['age']).agg('sum')['suicides_no'].keys())
y= list(data.groupby(['age']).agg('sum')['suicides_no'].values)
plt.figure(figsize=(4,4))
plt.barh(x,y)
plt.show


# ## Checking count of both genders in all

# In[ ]:


sns.countplot(x='sex', data=data)


# ## Country+sex suicide rate analysis

# In[ ]:


count_sex_total = data.groupby(['country','sex']).agg('sum')['suicides_no'].reset_index()

plt.figure(figsize=(8,8))
g = sns.FacetGrid(count_sex_total, col="country", height=6, size=4, col_wrap=6)
g.map(plt.bar, "sex", "suicides_no")
plt.show()


# >> In Brazil, France, Germany, Mexico, Poland, Russian Federation, Ukraine, United States huge difference between female and male
# 

# ## Count of suicides by gender

# In[ ]:


plt.figure(figsize=(8,4))
x = list(data.groupby(['sex']).agg('sum')['suicides_no'].keys())
y= list(data.groupby(['sex']).agg('sum')['suicides_no'].values)
sns.barplot(x, y, data=data)
plt.show()


# ## Age+Gender suicide rate analysis

# In[ ]:


age_gender_suicides_sum = data.groupby(['sex','age']).agg('sum')['suicides_no'].reset_index()


# In[ ]:


age_gender_suicides_sum


# In[ ]:


plt.figure(figsize=(8,8))
g = sns.FacetGrid(age_gender_suicides_sum, col='age', height= 6, size=4, col_wrap=3)
g.map(plt.bar, "sex", "suicides_no")
g.set_xticklabels(rotation=30,fontsize=10)


# ## Generation Analysis

# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='generation', data=data, order = data['generation'].value_counts().index)
plt.show()


# ## Generation + Age analysis

# In[ ]:


plt.figure(figsize=(8,6))
sns.countplot(x='generation', data=data, order = data['generation'].value_counts().index, hue='age')
plt.show()


# Generation with age doesn't reveal much as generations category are created based on age distribution itself.
# 

# Generation Analysis -:
# 
# > Generation Z : small children(5-14)
# 
# > Millenials : Youngesters + smallchildren (5-34)
# 
# > Boomers :  Youngsters + Middle_age (25-74)
# 
# > Silent: Middle_age (35-75)
# 
# > Generation X : Middle_age + youngesters + smallchildren (5-54)
# 
# > G.I Generation : Old (55-75)

# In[ ]:


data.groupby(['generation']).agg(['sum'])['suicides_no']


# In[ ]:


plt.figure(figsize=(8,4))
x = list(data.groupby(['generation']).agg('sum')['suicides_no'].keys())
y= list(data.groupby(['generation']).agg('sum')['suicides_no'].values)
sns.barplot(x, y, data=data)
plt.show()


# So, we can conclude from above graph -:
# 
# 1. Generation Z -: small_children has very low chance to suicide
# 2. G.I Generation -: old people do but comparatively less
# 3. Millenials -: youngsters alone not much
# 4. Boomers, Silen -: middle age and youngsters are doing more suicides
# 5. generation_x -: all (small children, youngsters, middle age) so has high number.
# 

# In[ ]:


generation_gender_sum = data.groupby(['generation', 'sex']).agg('sum')['suicides_no'].reset_index()

plt.figure(figsize= (12,4))
g = sns.FacetGrid(generation_gender_sum, col='sex', height=6, size=4)
g.map(plt.bar, "generation", "suicides_no")
g.set_xticklabels(rotation=90)


# In[ ]:


data.columns


# In[ ]:


for e in set(data['country']):
    plt.figure(figsize= (8,8))
    ax = sns.barplot(x="country-year", y="gdp_per_capita ($)", data=data[data['country']==e])
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.show()
    

