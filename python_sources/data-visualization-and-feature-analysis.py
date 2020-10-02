#!/usr/bin/env python
# coding: utf-8

# In[177]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as st
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
plt.style.use('fivethirtyeight')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[178]:


df = pd.read_csv('../input/dengue-dataset.csv')


# In[179]:


df.info()


# 17 years of data with monthly frequency. 
# One column has missing values, we need to see whether we can fill those without changing the underlying stature of the data

# In[180]:


df.head(5)


# I will change the column headings to English

# In[181]:


cols = ['Date', 'Confirmed_cases', 'Rain', 'Mean_temp', 'Min_temp', 'Max_temp']
df.columns = cols


# Feature Engineering

# In[182]:


# Extract the year and month from the date and separate them as different columns. 
df = pd.concat([pd.DataFrame([each[:2] for each in df['Date'].str.split('-').values.tolist()],
                             columns=['Year', 'Month']),df],axis=1)

# Convert the Date column to datetime
df.Date = df.Date.apply(lambda x : pd.to_datetime(x))

# Set the Date column as index
df.set_index('Date', inplace=True)

# Set the frequency of time series as Monthly
df = df.asfreq('MS')


# In[183]:


df.info()


# Now we will try to fill the missing values

# In[184]:


df.loc[df.Rain.isnull(),'Rain']


# In[185]:


missing_df = df[(df.Month == '06') | (df.Month == '07') | (df.Month == '08')]
missing_df.dropna(inplace=True)


# In[186]:


sns.boxplot(x = missing_df.Month, 
            y = missing_df.Rain)


# In[187]:


missing_df.groupby('Month')['Rain'].describe()


# Filling the missing values with the average doesn't seem to be a good idea here.
# Are there any relation between other available factors and rain?

# In[188]:


fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(missing_df[['Mean_temp', 'Min_temp', 'Max_temp', 'Rain']].corr(method='spearman'),
                       ax=ax,cmap='coolwarm',
                       annot=True)


# Maximum temperature has a slight correlation with the amount of rain fallen, 
# so if we average the instances of rain from corresponding months with comparable temperature 
# , it could be a better approximation. Before we do that, we will just check whether there is a correlation between rain fallen on subsequent months

# In[189]:


from statsmodels.graphics.tsaplots import plot_pacf


# In[190]:


plot_pacf(df['Rain'].dropna(), lags=24)


# Correlation on first lag is noticeable, and then there are quarterly and yearly seasonal correlations, but it is not as strong as the correlation noticed on the first lag, so if we just forward fill the missing values, it could also be considered as a good approximation and it is easy to implement as well.

# In[191]:


df['Rain'].fillna(method='ffill',inplace=True)


# Now we will check the yearly trend of the data

# In[192]:


df_yearly = df.groupby('Year')['Confirmed_cases'].agg('sum')


# In[193]:


ax = df_yearly.plot(kind='bar',
                    title = 'Cases by Year',figsize=(15,8) )
ax.set_xlabel("Year")
ax.set_ylabel("Cases")
ax.axhline(df.Confirmed_cases.mean(), color= 'r', ls = 'dotted', lw =1.5)
plo = (df_yearly.pct_change())*100
for (i, v) , val in zip(enumerate(plo), df_yearly):
    ax.text(i-.2, val + 500, val, color='black', fontsize=11)
    if i != 0:
        if v > 0:
            colr = 'green'
        else:
            colr = 'red'
        ax.text(i-.3, val + 2000, str(int(v)) + '%', color=colr, fontsize=11)


# There are no consistent upward or downward trends, each year the cases vary randomly.

# Now, monthly trend

# In[194]:


df_year_month = df.groupby(['Year','Month'], as_index=False).agg('sum')


# In[195]:


cross= pd.crosstab(df_year_month['Year'], df_year_month['Month'], df_year_month['Confirmed_cases'],aggfunc='sum')


# In[196]:


fig, ax = plt.subplots(figsize=(10,5))
plt.title("Cases - Yearly %")
sns.heatmap((cross.div(cross.sum(axis=1), axis=0)*100),annot=True)


# First half the year seems relatively busy with most number of cases reported on March-May time frame, For example, in 2014, 92% of that year's cases were reported on this time frame. 
# 
# Does any other feature available in the dataset follow similar pattern? 

# In[197]:


cross= pd.crosstab(df_year_month['Year'], df_year_month['Month'], df_year_month['Mean_temp'],aggfunc='sum')


# In[198]:


fig, ax = plt.subplots(figsize=(10,5))
plt.title("Mean Temperature - Yearly")
sns.heatmap(cross,annot=True)


# Mean temperature during the first four months are high, so is the last 4 months. If there was a direct relation, then the last 4 months of the year also should have reported high number of incidents, which is not the case here

# In[199]:


cross= pd.crosstab(df_year_month['Year'], df_year_month['Month'], df_year_month['Rain'],aggfunc='sum')


# In[200]:


fig, ax = plt.subplots(figsize=(10,5))
plt.title("Rain - Yearly %")
sns.heatmap((cross.div(cross.sum(axis=1), axis=0)*100),annot=True)


# Rain alone is not a factor, could the combined effect of rain and the temperature influence the outbreak ?  also there is a possibility of  lagged effect as well. We will check the individual lagged effects, first

# In[201]:


for i in range(0,5):
    print('lag ' + str(i) + ' = ' + str(df['Confirmed_cases'].corr(df['Rain'].shift(i))))


# It is evident that lag 2 has higher absolute correlation than the 0th lag, so rain could possibly have a lagged effect on the dengue cases

# In[202]:


for i in range(0,5):
    print('lag ' + str(i) + ' = ' + str(df['Confirmed_cases'].corr(df['Mean_temp'].shift(i))))


# Lag 2 and 3 of Mean temperature show better correlation to dengue cases than the 0th lag

# Somehow the temperature and rain fall from the previous months have a correlation to the dengue cases, this might have something to do with the lifecycle of mosquitos  

# We will build a simple regression model and analyse the coefficients of the model to better undestand the correlations we have noticed above

# In[207]:


df_lag = df.copy()


# In[208]:


df_lag['Rain'] = df_lag.Rain.shift(2)
df_lag['Mean_temp'] = df_lag.Mean_temp.shift(3)


# In[209]:


df_lag.dropna(axis=0,inplace=True)


# In[210]:


df_lag.info()


# In[211]:


import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# We will build several OLS models and compare their influence

# In[213]:


least_square1 = smf.ols('Confirmed_cases ~ Rain -1', data=df_lag).fit()
least_square2 = smf.ols('Confirmed_cases ~ Mean_temp -1', data=df_lag).fit()
least_square3 = smf.ols('Confirmed_cases ~ Rain + Mean_temp -1', data=df_lag).fit()
least_square4 = smf.ols('Confirmed_cases ~ Rain*Mean_temp -1', data=df_lag).fit()


# In[214]:


anova_lm(least_square1, least_square2, least_square3, least_square4)


# In[215]:


least_square4.summary()


# The model with Rain and Mean_temp, interacting together should give better result, but Mean_temp alone should also work as the P values suggest

# In[216]:


least_square5 = smf.ols('Confirmed_cases ~ Month -1', data=df_lag).fit()


# In[217]:


least_square5.summary()


# As we have noticed before from heat maps, the model gives high importance to March, April and May.

# In[218]:


anova_lm(least_square4, least_square5)


# In[224]:


least_square6 = smf.ols('Confirmed_cases ~ Rain*Mean_temp + Month-1', data=df_lag).fit()


# In[225]:


least_square6.summary()


# In[226]:


anova_lm(least_square5, least_square6)


# So a model with Interactions between Rain and Mean temperature combined with the Month should be able to explain 24.5% of the variability

# In[ ]:




