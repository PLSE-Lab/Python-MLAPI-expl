#!/usr/bin/env python
# coding: utf-8

# # Sberbank Russian Housing Market - Exploratory Data Analysis 2
# 
# You can see the first part of my analysis here: 
# https://www.kaggle.com/rikhard/eda-01-missing-data-structure-and-regressors/notebook
# 
# In this kernel, we will look at the macro data. At the beginning, we will follow the same structure that we used in our previous analysis.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Load train data
df = pd.read_csv('../input/macro.csv')


# ## Data description
# 
# First, let's see the size and types of data we have:

# In[ ]:


print(df.columns.tolist())
print('\nNumber of columns on train data:',len(df.columns))
print('\nNumber of data points:',len(df))
print('\nNumber of unique timestamp data points:',len(df['timestamp'].unique()))
print('\nData types:',df.dtypes.unique())


# It seems we have non numerical data here. Let's take a look at it:

# In[ ]:


df.select_dtypes(include=['O']).columns.tolist()


# In[ ]:


df.select_dtypes(include=['O']).dropna().head()


# ### Some data needs to be processed
# 
# These columns should also be numeric, so there must be some bad data in them. If we try to use the pandas method 'pd.to_numeric()', we will find that it can't parse the strings in this columns and if we try to force the conversion, it will return all NaNs in the three columns. The problem is that those numbers have the symbol "," instead of "." so the parser doesn't recognize them as numbers. The easiest way to overcome this problem is to replace the commas per dots and then use the pandas method to change the column dtype:

# In[ ]:


object_cols = df.select_dtypes(include=['O']).columns.tolist()[1:]
for col in object_cols:
    df[col] = df[col].apply(lambda s: str(s).replace(',','.'))
df[object_cols] = df[object_cols].apply(pd.to_numeric, errors='coerce')
df[object_cols].dropna().head()


# In[ ]:


df.select_dtypes(include=['O']).columns.tolist()


# Problem solved. Now let's see how many NaNs are there in the macro data:

# In[ ]:


# Get the number of NaN's for each column, discarding those with zero NaN's
ranking = df.loc[:,df.isnull().any()].isnull().sum().sort_values()
# Turn into %
x = ranking.values/len(df)

# Plot bar chart
index = np.arange(len(ranking))
plt.bar(index, x)
plt.xlabel('Features')
plt.ylabel('% NaN observations')
plt.title('% of null data points for each feature')
plt.show()

print('Features:',ranking.index.tolist())
print('\nNumber of columns which have any NaN:',df.isnull().any().sum(),'/',len(df.columns))
print('\nNumber of rows which have any NaN:',df.isnull().any(axis=1).sum(),'/',len(df))


# There is a lot of NaNs in the macro data. But that is not the only problem:

# In[ ]:


df.iloc[1000:1010,:10]


# ## Separating macro data depending on its period
# 
# We have mixed periods of data: annual, quarterly, monthly, daily ... And the main problem is not NaNs, but duplicate data. We will have to write an algorythm to separate the data depending on its period. We will begin creating new time features:

# In[ ]:


df['year'] = df['timestamp'].apply(lambda f: f.split('-')[0])
df['month'] = df['timestamp'].apply(lambda f: f.split('-')[1])
df['day'] = df['timestamp'].apply(lambda f: f.split('-')[2])
df['quarter'] = np.floor((df['month'].values.astype('int')-1)/3).astype('int')+1
df['year-quarter'] = df['year'] +'-Q' + df['quarter'].astype('str')
df['year-month'] = df['year'] +'-'+ df['month']
del df['quarter']


# Next we create a function that calculates the period of a data series:

# In[ ]:


def get_periods(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    var = df.columns[1]
    df['shift'] = df[var].shift()
    df = df.dropna()
    index = df[df[var]-df['shift']!=0].index
    if len(df.loc[:,var].unique())>3:
        return df.loc[index,'timestamp'].diff().mean().round('D')
    else: return -1


# In[ ]:


periods = list()
for i,col in enumerate(df.columns[1:-5]):
    periods.append(get_periods(df[['timestamp',col]].copy()))
    print(i,col,periods[i])


# In[ ]:


set(periods)


# Yet we have to write another algorythm to purify that set and get the final time categories for macro data:

# In[ ]:


final_periods = dict()
for col, period in zip(df.columns[1:-5], periods):
    if period == -1: continue
    elif period <= pd.Timedelta('2 days'): final_periods[col] = 1
    elif period <= pd.Timedelta('35 days'): final_periods[col] = 30
    elif period <= pd.Timedelta('100 days'): final_periods[col] = 90
    elif period <= pd.Timedelta('370 days'): final_periods[col] = 365
    
print('Number of columns of interest:',len(final_periods))


# ## Plotting the macro data
# 
# Now that we have preprocessed the data, we can plot all the macro features:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
for key, value in final_periods.items():
    resume = pd.DataFrame(columns=[key,'mean'])
    if value == 1:
        resume[key] = df[key]
        resume['mean'] = resume[key].dropna().rolling(30,center=True).mean()
    elif value == 30:
        resume[key] = df.groupby('year-month').agg('mean')[key]
        resume['mean'] = resume[key].dropna().rolling(12,center=True).mean()
    elif value == 90:
        resume[key] = df.groupby('year-quarter').agg('mean')[key]
        resume['mean'] = resume[key].dropna().rolling(4,center=True).mean()
    elif value == 365:
        resume[key] = df.groupby('year').agg('mean')[key]
    resume.plot()
    


# 
# This is all for the moment. If you liked this analysis, please upvote! Thank's!
