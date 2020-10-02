#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 1. Exploring dataset

# In[ ]:


data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

print('Number of features: %s' %data.shape[1])
print('Number of examples: %s' %data.shape[0])


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# Dataset free from null values

data.isnull().sum()


# In[ ]:


# Printing unique values

for col in data[['year', 'state', 'month']]:
    print('Unique values in column: %s' %col)
    print(data[col].unique())
    print('\n')

print('Number of unique values: ')
print(data[['year', 'state', 'month']].nunique())


# In[ ]:


# Setting months to english

english_month = ['january','feburary','march','april','may','june','july','august','september','october','november','december']
data_month = list(data['month'].unique())

month_dict = dict(zip(data_month, english_month))
month_dict


# In[ ]:


data.month = data['month'].map(month_dict)


# 2. Evaluations

# In[ ]:


# Evaluation 1 - general values for dataset

fires_number = pd.DataFrame(data['number'].describe())
fires_number.columns = ['Fires']
fires_number['Stats'] = fires_number.index
fires_number.reset_index(inplace=True, drop=True)
fires_number


# In[ ]:


# Evaluation 2 - number of fires in every month in every state

data1 = pd.DataFrame(data.groupby(['month', 'state'])['number'].mean())
data1.head()


# 3. Visualisations

# In[ ]:


# Visualisation 1 - number of fires in years
data2 = pd.DataFrame(data.groupby('year')['number'].sum().reset_index())

plt.figure(figsize=(15,8))
sns.lineplot(x='year', y='number', data=data2, color='red', lw=3)
plt.xlabel('Year')
plt.ylabel('Number of fires')
plt.title('Number of fres in Brazil per year', fontsize=15)
plt.xlim(1998,2017)
plt.xticks(np.arange(1998, 2018, 1),fontsize=12)
plt.yticks(fontsize=12)


# In[ ]:


# Visualisation 2 - fires related to states

data3 = pd.DataFrame(data.groupby('state')['number'].sum().sort_values(ascending=False).reset_index())
data3['state'].iloc[12] = 'Para'
plt.figure(figsize=(15,8))
sns.barplot(data=data3, x='state', y='number', palette='autumn')
plt.title('Number of fires in Brazil per state', fontsize=15)
plt.xlabel('States')
plt.ylabel('Number of fires')
plt.xticks(rotation=80)


# In[ ]:


# Visualisation 3 - fires per month

data4 = pd.DataFrame(data.groupby('month')['number'].sum().sort_values(ascending=False).reset_index())

plt.figure(figsize=(15,8))
sns.barplot(data=data4, x='month', y='number', palette='autumn')
plt.title('Number of fires in Brazil per month', fontsize=15)
plt.xlabel('Month')
plt.ylabel('Number of fires')
plt.xticks(rotation=80)


# In[ ]:


# Visualisation 4 - number of fires in Amazonas

data5 = pd.DataFrame(data[data['state']=='Amazonas'])
data5 = data5.groupby('year')['number'].sum().reset_index()

plt.figure(figsize=(15,8))
sns.lineplot(x='year', y='number', data=data5, lw=3, color='red')
plt.title('Number of fires in Amazonas', fontsize=15)
plt.xlabel('Year')
plt.ylabel('Number of fires')
plt.xticks(np.arange(1997,2017,1), rotation=80)
plt.xlim(1998,2018)


# In[ ]:


# Visualisation 5 - total fires in 5 top
# Rio, Paraiba, Mato Grosso, Alagoas

df1 = pd.DataFrame(data=data[data['state'] =='Rio'])
df2 = pd.DataFrame(data=data[data['state'] =='Paraiba'])
df3 = pd.DataFrame(data=data[data['state'] =='Mato Grosso'])
df4 = pd.DataFrame(data=data[data['state'] =='Alagoas'])

plt.figure(figsize=(15,8))

df_list = [df1, df2, df3, df4]
df_group = []
for x in df_list:
    x.groupby('year')['number'].sum().reset_index()
    df_group.append(x)

for x in df_group:
    sns.lineplot(x='year', y='number', data=x, lw=1, label=x['state'].iloc[0])
    plt.title('Fires in top 5 states', fontsize=15)
    plt.xlabel('Year')
    plt.ylabel('Number of fires')
    plt.xticks(np.arange(1997,2017,1), rotation=80)
    plt.xlim(1998,2017)

plt.legend(fontsize=13)


# In[ ]:


# Visualisation 6 - pivot table

data_pivot = data.pivot_table(values='number', index='year', columns='month', aggfunc=np.sum)
data_pivot = data_pivot.loc[:,['january', 'feburary', 'march', 'april', 'may', 'june', 'july', 'august','september', 'october', 'november', 'december']]

plt.figure(figsize=(15,8))
sns.heatmap(data_pivot, linewidths=0.05, vmax=9000, cmap='Oranges', fmt="1.0f", annot=True)
plt.title('Heatmap of number of fires in states in every month in years', fontsize=15)
plt.xlabel('Month')
plt.ylabel('Year')


# In[ ]:




