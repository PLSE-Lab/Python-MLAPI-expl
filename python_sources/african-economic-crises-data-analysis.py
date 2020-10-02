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


# <h1>Africa Economic, Banking and Systemic Crises Dataset Data Analysis</h1>
# <h3>Dataset Context</h3>
# <p>This dataset is a derivative of Reinhart et. al's Global Financial Stability dataset which can be found online at:<br><a href='https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx'>Data Source</a><br>The dataset will be valuable to those who seek to understand the dynamics of financial stability within the African context.</p>
# <h3>Dataset Content:</h3>
# <p>The dataset specifically focuses on the Banking, Debt, Financial, Inflation and Systemic Crises that occurred, from 1860 to 2014, in 13 African countries, including: Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia and Zimbabwe.</p>
# 

# In[ ]:


african_crises_df=pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
print('Dataset rows: ',african_crises_df.shape[0])
print('Dataset columns: ',african_crises_df.shape[1])
print('column names:\n',african_crises_df.columns)


# <h3>Dataset Description</h3>
# <ul>
#     <li>case: A number which denotes a specific country</li>
#     <li>cc3: A three letter country code</li>
#     <li>country: The name of the country</li>
#     <li>year: The year of the observation</li>
#     <li>systemic_crisis: "0" means that no systemic crisis occurred in the year and "1" means that a systemic crisis occurred in the year.</li>
#     <li>exch_usd: The exchange rate of the country vis-a-vis the USD</li>
#     <li>domestic_debt_in_default: "0" means that no sovereign domestic debt default occurred in the year and "1" means that a sovereign domestic debt default occurred in the year</li>
#     <li>sovereign_external_debt_default: "0" means that no sovereign external debt default occurred in the year and "1" means that a sovereign external debt default occurred in the year</li>
#     <li>gdp_weighted_default: The total debt in default vis-a-vis the GDP</li>
#     <li>inflation_annual_cpi: The annual CPI Inflation rate</li>
#     <li>independence: "0" means "no independence" and "1" means "independence"</li>
#     <li>currency_crises: "0" means that no currency crisis occurred in the year and "1" means that a currency crisis occurred in the year</li>
#     <li>inflation_crises: "0" means that no inflation crisis occurred in the year and "1" means that an inflation crisis occurred in the year</li>
#     <li>banking_crisis: "no_crisis" means that no banking crisis occurred in the year and "crisis" means that a banking crisis occurred in the year</li>
# </ul>
# 

# In[ ]:


def explore_df(df):
    print('Data type of each column:\n')
    print(african_crises_df.dtypes)
    cer_cols=african_crises_df.select_dtypes(include=['object','int']).columns
    for i in cer_cols:
            print('Column name: ',i,'\n')
            print(african_crises_df[i].value_counts().sort_values())
    print('missing values in each column\n')
    print(african_crises_df.isnull().sum())
explore_df(african_crises_df)


# <h1>Data Cleaning</h1>

# <ul>
#     <li>first we are going to drop some columns </li>
#     <li>change data type values in banking_crises column to 0s and 1s</li>
#     <li>In currency_crises column there is 4 rows its value is 2 but this is error because this column must include 0s and 1s</li>
# <ul>

# In[ ]:


cols_to_drop=['case','cc3']
cleaned_df=african_crises_df.drop(cols_to_drop,axis=1)
labels=pd.Categorical(cleaned_df['banking_crisis'])
cleaned_df['banking_crisis']=labels.codes
cleaned_df['currency_crises']=cleaned_df['currency_crises'].replace(2,np.nan)
cleaned_df=cleaned_df.dropna()


# In[ ]:


cleaned_df['currency_crises']=cleaned_df['currency_crises'].astype(int)


# <h1>EDA</h1>

# In[ ]:


apply_stats_cols=['exch_usd','gdp_weighted_default','inflation_annual_cpi']
cleaned_df[apply_stats_cols].describe()


# In[ ]:


cleaned_df.corr()


# In[ ]:


sns.heatmap(cleaned_df.corr())


# <p>note: I think that correlation here means nothing because most of dataset are categories</p> 

# <p>first let's examine the exchange rate distribution for each country vis a vis USD (1950-2014)</p> 

# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(25,10))
countries=cleaned_df['country'].unique().tolist()
count=1
for i in countries:
    plt.subplot(3,5,count)
    count+=1
    sns.lineplot(cleaned_df[cleaned_df.country==i]['year'],
                 cleaned_df[cleaned_df.country==i]['exch_usd'],
                 label=i,
                 color='black')
    plt.scatter(cleaned_df[cleaned_df.country==i]['year'],
                cleaned_df[cleaned_df.country==i]['exch_usd'],
                color='black',
                s=28)
    plt.plot([np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year']),
              np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year'])],
             [0,
              np.max(cleaned_df[cleaned_df.country==i]['exch_usd'])],
             color='green',
             linestyle='dotted',
             alpha=0.8)
    plt.text(np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year']),
             np.max(cleaned_df[cleaned_df.country==i]['exch_usd'])/2,
             'Independence',
             rotation=-90)
    plt.scatter(x=np.min(cleaned_df[np.logical_and(cleaned_df.country==i,cleaned_df.independence==1)]['year']),
                y=0,
                s=50)
    plt.title(i)
plt.show()


# <p>number of times that occurs a sovereign domestic debt default for each country(1950-2014)</p>

# In[ ]:


plt.style.use('fivethirtyeight')
fig=plt.figure(figsize=(25,10))
countries=cleaned_df['country'].unique().tolist()
lst_len=len(countries)
for i in range(lst_len-1):
    ax=fig.add_subplot(2,6,i+1)
    c=cleaned_df[cleaned_df['country']==countries[i]]['domestic_debt_in_default'].value_counts()
    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)
    ax.set_title(countries[i])
    plt.legend(loc='best')
plt.show()


# <p>Angola has 10 sovereign domestic debt default occured</p>

# <p>number of times that occurs a sovereign external debt default for each country(1950-2014)</p>

# In[ ]:


plt.style.use('fivethirtyeight')
fig=plt.figure(figsize=(25,10))
countries=cleaned_df['country'].unique().tolist()
lst_len=len(countries)
for i in range(lst_len-1):
    ax=fig.add_subplot(2,6,i+1)
    c=cleaned_df[cleaned_df['country']==countries[i]]['sovereign_external_debt_default'].value_counts()
    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)
    ax.set_title(countries[i])
    plt.legend(loc='best')
plt.show()


# In[ ]:


fig=plt.figure(figsize=(25,10))
countries=cleaned_df['country'].unique().tolist()
lst_len=len(countries)
for i in range(lst_len-1):
    ax=fig.add_subplot(2,6,i+1)
    c=cleaned_df[cleaned_df['country']==countries[i]]['systemic_crisis'].value_counts()
    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)
    ax.set_title(countries[i])
    plt.legend(loc='best')
plt.show()


# In[ ]:


fig=plt.figure(figsize=(25,10))
countries=cleaned_df['country'].unique().tolist()
lst_len=len(countries)
for i in range(lst_len-1):
    ax=fig.add_subplot(2,6,i+1)
    c=cleaned_df[cleaned_df['country']==countries[i]]['banking_crisis'].value_counts()
    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)
    ax.set_title(countries[i])
    plt.legend(loc='best')
plt.show()


# In[ ]:


fig=plt.figure(figsize=(25,10))
countries=cleaned_df['country'].unique().tolist()
lst_len=len(countries)
for i in range(lst_len-1):
    ax=fig.add_subplot(2,6,i+1)
    c=cleaned_df[cleaned_df['country']==countries[i]]['currency_crises'].value_counts()
    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)
    ax.set_title(countries[i])
    plt.legend(loc='best')
plt.show()


# In[ ]:


fig=plt.figure(figsize=(25,10))
countries=cleaned_df['country'].unique().tolist()
lst_len=len(countries)
for i in range(lst_len-1):
    ax=fig.add_subplot(2,6,i+1)
    c=cleaned_df[cleaned_df['country']==countries[i]]['inflation_crises'].value_counts()
    ax.bar(c.index,c.tolist(),color=['yellow','green'],width=0.4)
    ax.set_title(countries[i])
    plt.legend(loc='best')
plt.show()

