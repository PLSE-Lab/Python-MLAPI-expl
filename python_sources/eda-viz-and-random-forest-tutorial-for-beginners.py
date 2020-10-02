#!/usr/bin/env python
# coding: utf-8

# 1. [Read in Data and Import Essential Libraries ](#read)
# 2. [Looking at Data](#lookAtData)
# 3. [Exploratory Data Analysis (EDA)](#EDA)
#     - 3.1 [Countries with highest "suicide/100k population rate" for each age group](#highestsuiciderate)
#     - 3.2 [global suicide rate by gender and by age groups over time](#globalsuiciderateovertime)
#     - 3.3 [differences between countries where people are very likely to commit suicide and countries that are the opposite](#highsuicidevslowsuicide)
#     - 3.4 [Coutries with Aging Population](#agingpop)
#     - 3.5 [countries have high proportion of Millenials?](#highpropmillenials)
# 4. [Random Forest Modelling](#rfm)
#     - 4.1 [Feature Importance](#featureimport)
# 

# <a id='read'></a>
# # Read in Data and Import Essential Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read in Data
df = pd.read_csv("../input/master.csv")


# <a id='lookAtData'></a> 
# # Look at Data

# In[ ]:


# Looking at first five rows of the data
df.head()


# In[ ]:


# Basic summary statistics of data
df.describe()


# In[ ]:


# Number of rows and columns of the dataset
df.shape


# In[ ]:


# Checking number of missing data for each column
df.isnull().sum()


# The "HDI" column seems to be the only feature that has a lot of missing values. As a side note, HDI stands for Human Development Index. A country scores a higher HDI when the lifespan is higher, the education level is higher, and the GNI per capita is higher.

# <a id='EDA'></a> 
# # Exploratory Data Analysis

# In[ ]:


# Some aesthetic setting
plt.style.use('ggplot')
sns.set_style("whitegrid")


# <a id='highestsuiciderate'></a>
# ## Which country has, on average, the highest "suicide/100k population rate" for each age group?

# In[ ]:


# Top 10 countries with highest average suicide rate for each age group
age_groups = df.age.unique().tolist()
for age_group in age_groups: 
    print("Top 10 countries with highest average suicide rate for", age_group)
    print(pd.DataFrame(df.groupby(['country','age'])['suicides/100k pop'].mean()).reset_index().    sort_values(['age','suicides/100k pop']).groupby('age').get_group(age_group).    sort_values('suicides/100k pop', ascending=False).head(10))


# Central Asian Countries, Eastern European countries and Sri Lanka consistenly appear in the list across different age groups although there are some additions of countries for only certain age groups (e.g. For 75+ years, addition of South Korea)

# <a id="globalsuiciderateovertime"></a>
# ## Is there a change in global suicide rate by gender and by age groups over time?

# In[ ]:


f, (ax1,ax2) = plt.subplots(2,1,sharex=True, sharey=True, figsize=(20,8))

# Male
sns.pointplot(x='year',y='suicides/100k pop',
data=pd.DataFrame(df[df.sex=='male'].groupby(['year','age'])['suicides/100k pop'].sum()).reset_index().\
sort_values('age'),hue='age',ax=ax1)

# Female
ax2 = sns.pointplot(x='year',y='suicides/100k pop',
data=pd.DataFrame(df[df.sex=='female'].groupby(['year','age'])['suicides/100k pop'].sum()).reset_index().\
sort_values('age'), hue='age',ax=ax2)


# - Except for 5-14 years age group, sum of male suicide rate around the world is higher than that of females across different age groups
# - There was some major spike in sum of suicide rate for males in the early 1990s 
# - For both male and females, the total suicide rates around the world have plummeted considerably in the recent few years

# <a id="highsuicidevslowsuicide"></a>
# ## What are some differences between countries where people are very likely to commit suicide and countries that are the opposite?

# We have first identify (or define) which countries have people who are very likely to commit suicide. I will define them as countries which are included in the list for top10 countries for all 4 metrics: mean suicide number, total suicide number, mean suicides/100k pop and total suicides/100k pop. The countries less prone to suicide will be the countries included in the list for bottom 10 countries for the same 4 metrics.

# In[ ]:


pd.Series(df.groupby(['country']).suicides_no.mean().sort_values(ascending=False).head(10).index.tolist() + df.groupby(['country']).suicides_no.sum().sort_values(ascending=False).head(10).index.tolist() + df.groupby(['country'])['suicides/100k pop'].mean().sort_values(ascending=False).head(10).index.tolist() + df.groupby(['country'])['suicides/100k pop'].sum().sort_values(ascending=False).head(10).index.tolist()).value_counts()


# Ukraine and Russia are the countries most prone to suicide

# In[ ]:


pd.Series(df.groupby(['country']).suicides_no.mean().sort_values().head(10).index.tolist() + df.groupby(['country']).suicides_no.sum().sort_values().head(10).index.tolist() + df.groupby(['country'])['suicides/100k pop'].mean().sort_values().head(10).index.tolist() + df.groupby(['country'])['suicides/100k pop'].sum().sort_values().head(10).index.tolist()).value_counts()


# Maldives and Saint Kitts and Nevis are the countries with the happiest(?) people less likely to commit suicide

# Let's now compare these two sets of countries and how they are different

# In[ ]:


# High suicide rate countries (hsrc)
df_hsrc = df[df.country.isin(['Ukraine','Russian Federation'])]

# Low suicide rate countries (lsrc)
df_lsrc = df[df.country.isin(['Maldives','Saint Kitts and Nevis'])]


# In[ ]:


# High suicide rate countries (hsrc) gdp_per_capita summary statistics
df_hsrc[['country','gdp_per_capita ($)']].groupby('country').describe()


# In[ ]:


# Low suicide rate countries (lsrc) gdp_per_capita summary statistics
df_lsrc[['country','gdp_per_capita ($)']].groupby('country').describe()


# In[ ]:


# gdp_per_capita summary statistics of all 4 countries combined
df_hsrc[['country','gdp_per_capita ($)']].groupby('country').describe().append(df_lsrc[['country','gdp_per_capita ($)']].groupby('country').describe())


# There is not much information about each of the country, so I referred the International Labour Organization(ILO) Database for extra info.

# [International Labour Organization(ILO) Database](https://www.ilo.org/ilostat/faces/oracle/webcenter/portalapp/pagehierarchy/Page21.jspx?_afrLoop=1118754007519581&_afrWindowMode=0&_afrWindowId=null#!%40%40%3F_afrWindowId%3Dnull%26_afrLoop%3D1118754007519581%26_afrWindowMode%3D0%26_adf.ctrl-state%3D11sex41g62_13)
# 
# - Maldives
#     * Labor force participation rate 47.3%
#     * Labor force participation rate men 73.2%
#     * Labor force participation rate women 21.2%
#     * Employment to population ratio 42%
#     * Unemployment rate 11.2%
#     * Youth unemployment rate 17.6%
#     * share of employees working more than 48 hours per week 36.9%
#     * percentage of health care expenditure not financed by private households' out of pocket payments 20.6%
# 
# - Saint Kitts and Nevis
#     * Other metrics not available
#     * percentage of health care expenditure not financed by private households' out of pocket payments 58.2%
# 
# - Russia 
#     * Labor force participation rate 62.8%
#     * Labor force participation rate men 71.3%
#     * Labor force participation rate women 55.7%
#     * Employment to population ratio 59.5%
#     * Unemployment rate 5.2%
#     * Youth unemployment rate 16.3%
#     * share of employees working more than 48 hours per week 2.2%
#     * percentage of health care expenditure not financed by private households' out of pocket payments 64.6%
# 
# - Ukraine 
#     * Labor force participation rate 62.0%
#     * Labor force participation rate men 62.0%
#     * Labor force participation rate women 69.0%
#     * Employment to population ratio 56.7%
#     * Unemployment rate 9.5%
#     * Youth unemployment rate 18.9%
#     * share of employees working more than 48 hours per week 3.3%
#     * percentage of health care expenditure not financed by private households' out of pocket payments 58.5%

# Of course, we selected only two countries from each group, so there is nothing to conclude about the causality or even the correlation between various socioeconomic factors (e.g unemployment rate, HDI etc) and likelihood of citizens of that country committing suicide. But through EDA like this, we can at least get a sneak peek into what kind of attributes some of the high/low suicide rate countries have. At least for these two sets of countries, high suicide rate countries (Russia, Ukraine) seem to have overwhelmingly better records for all socioeconomic factors (e.g. lower unemployment rate, higher labor participation rate, lower share of unemployees working over time, higher percentage of health care expenditure not financed by private households' out of pocket payments etc.) than low suicide rate countries (e.g. Maldives, Saint Kitts and Nevis). Maybe the cause of higher suicide rate lies in other factors?

# <a id="agingpop"></a>
# ## Which countries having an aging population in 2014?

# In[ ]:


# Proportion of citizens who are 55 years old or older (Top 10)
df[df.year==2014].sort_values('age').groupby(['country','age']).population.sum().groupby('country').apply(lambda x: (x[3]+x[4])/sum(x)).sort_values(ascending=False).head(10).plot('bar')


# We see a lot of European countries are topping the list with the exception of Japan. Of course, this was 5 years ago, so things are different now (e.g. more Asian countries experiencing population aging etc)

# <a id="highpropmillenials"></a>
# ## Which countries have high proportion of Millenials? (YOUNG countries~~)

# In[ ]:


# Proportion of Millenials of each country in 2014 (Top 10)
df[df.year==2014].groupby(['country','generation']).population.sum().groupby('country').apply(lambda x: x[3]*100/sum(x)).sort_values(ascending=False).head(10).plot('bar')


# Contrary to the list of coutries with high proportions of elderly citizens, we see more countries from Asia and Central/South America from this list of countries with high proportions of Millenials

# In[ ]:


# Top 6 countries with highest proportions of millenials in 2014
top6_millenial_countries = df[df.year==2014][df[df.year==2014].country.isin(df[df.year==2014].groupby(['country','generation']).population.sum().groupby('country').apply(lambda x: x[3]*100/sum(x)).sort_values(ascending=False).head(6).index.tolist())]


# In[ ]:


top6_millenial_countries.head()


# In[ ]:


# Percentage of each generation of top 6 countries with highest proportion of millenials
f, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(10,7))
axes=[(ax1,'Bahrain'),(ax2,'Grenada'),(ax3,'Guatemala'),(ax4,'Oman'),(ax5,'Qatar'),(ax6,'Uzbekistan')]

for ax in axes:
    ax[0].set_title(ax[1])
    ax[0].pie(top6_millenial_countries[top6_millenial_countries.country==ax[1]].groupby('generation').population.sum(),
        labels=['Boomers', 'Generation X', 'Generation Z', 'Millenials', 'Silent'], autopct='%1.1f%%', 
        startangle = 150)
    plt.axis('equal')
plt.tight_layout()


# <a id="rfm"></a>
# # Random Forest Modelling

# Let's say you want to be able to explain which factors are influential to the number of suicides. You can set the "number of suicides" as your target variable(y) and the remaining variables as your predictor variables(x). 

# Let's drop HDI for now since it has too many missing values

# In[ ]:


df.drop('HDI for year',axis=1, inplace=True)


# In[ ]:


# Checking for missing values
df.isnull().sum()


# Most of the algorithms spit out errors if you have nan values in your data. Thus, it is crucial for you too check if there are any missing values left and fill them in using various methods (e.g. filling in with mean/median value, using regression to extrapolate the missing values etc)

# Next important step is to make sure all the categorical variables are encoded so that all data values are numerical! Except for algorithms like Catboost, most algorithms in sklearn require users to change categorical variables into numbers

# In[ ]:


# Checking data types
df.info()


# In[ ]:


# Remove commas in strings
df[' gdp_for_year ($) '] = df[' gdp_for_year ($) '].str.replace(",", "")

# Change string to numbers
df[' gdp_for_year ($) '] = pd.to_numeric(df[' gdp_for_year ($) '])


# In[ ]:


# Encoding of cat variables with LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for x in ['country','sex','age','country-year','generation']:
    df[x] = le.fit_transform(df[x])


# Decision Tree is a tree that splits at different features of the dataset for every data point. If a certain feature was used more frequently to split and make a decision (e.g classification of a data point), it means that feature is a important one in determining the value or class of data points you want to predict. For this reason, trees and forest algorithms can be used to determine  feature importance. Refer to the below picture!

# ![](https://cdn-images-1.medium.com/max/1600/1*TlTzgt8I_5dUSbMZmRKyqQ.jpeg)

# Forest is basically a conglomerate of trees! Random forest algorithm combines numerous decisions trees and thereby reduces overfitting (a phenomenon where the model performs extremely well for training data but doesn't for unseen new data). Random Forest also works very well for both categorical and numerical variables!

# ![](https://cdn-images-1.medium.com/max/1000/1*i0o8mjFfCn-uD79-F1Cqkw.png)

# In[ ]:


# Split dataset into X and y
y = df[['suicides_no']].values
X = df.drop('suicides_no',axis=1).values


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


from sklearn.metrics import mean_squared_error, make_scorer
rfr = RandomForestRegressor(random_state=42)

# Average RMSE of 5 fold cross validation
cross_val_score(rfr, X, y, cv=5).mean()


# <a id="featureimport"></a>
# ## Feature Importance

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rfr.fit(X,y), random_state=42).fit(X, y)
eli5.show_weights(perm, feature_names = df.drop('suicides_no',axis=1).columns.tolist(),top=10)


# As you can see above, the features that are marked positive are features that have a positive influential relationship with the target variable(y). In this case, they are population and the suicide rate. Since we don't have that many informative & useful features in the first place, there is not much to get out from this but at least the results make sense because country will bigger population is more likely to have more frequent suicide occurences (simply because there are more peopel available to kill themselves).

# ## ** Please consider upvoting this kernel if you liked it :) **
