#!/usr/bin/env python
# coding: utf-8

# # Pharma Sales Time Series Analysis
# 
# ## Getting Started

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 
# ## Reading the Data

# In[ ]:


#reading the data
hourly = pd.read_csv('/kaggle/input/pharma-sales-data/saleshourly.csv')
daily = pd.read_csv('/kaggle/input/pharma-sales-data/salesdaily.csv')
weekly = pd.read_csv('/kaggle/input/pharma-sales-data/salesweekly.csv')
monthly = pd.read_csv('/kaggle/input/pharma-sales-data/salesmonthly.csv')


# ## Analysing Data Stucture

# In[ ]:


#function to print shape of a given data
def print_shape(data):
    print('Rows : ',data.shape[0])
    print('Columns : ',data.shape[1])


# In[ ]:


print_shape(hourly)
print_shape(daily)
print_shape(weekly)
print_shape(monthly)


# From the shape of monthly dataframe, we see that the data is of 70 months.

# In[ ]:


hourly.head(2)


# In[ ]:


daily.head(2)


# In[ ]:


weekly.head(2)


# In[ ]:


monthly.head(2)


# Notice that the format of datum column is different in hourly and monthly data and same in daily and weekly data.

# In[ ]:


#copy the data
hourly_original = hourly.copy()
daily_original = daily.copy()
weekly_original = weekly.copy()
monthly_original = monthly.copy()


# Let us now convert data type of datum column from object to datetime

# In[ ]:


#converting datatype of dates from object to Datetime
monthly['datum'] = pd.to_datetime(monthly['datum'], format= '%Y-%m-%d')
weekly['datum'] = pd.to_datetime(weekly['datum'], format= '%m/%d/%Y')
daily['datum'] = pd.to_datetime(daily['datum'], format= '%m/%d/%Y')
hourly['datum'] = pd.to_datetime(hourly['datum'], format= '%m/%d/%Y %H:%M')


# # Analysing Monthly Series
# 
# Firstly, let us analyse the monthly data and see what inferences can we draw from this data.

# In[ ]:


#import datetime for dates and time realted calculations
import datetime as dt


# Seperate year, month and day from the datum column

# In[ ]:


#extracting year from dates
monthly['year'] = monthly['datum'].dt.year


# In[ ]:


#extracting month from dates
monthly['month'] = monthly['datum'].dt.month


# In[ ]:


#extracting day from dates
monthly['day'] = monthly['datum'].dt.day


# In[ ]:


#set index equal to the dates which will help us in visualising the time series
monthly.set_index(monthly['datum'], inplace= True)


# In[ ]:


monthly.head(2)


# In[ ]:


#define a function to plot yearly sales of every category of drug.
def plot_yearly_sales(column):
    monthly.groupby('year')[column].mean().plot.bar()#calculating yearly sales using groupby
    plt.title(f'Yearly sales of {column}')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.show()


# In[ ]:


#plotting yearly sales of each drug category
for i in monthly.columns[1:9]:#drug categories are from 1 to 8 index
    plot_yearly_sales(i) 


# Analysing the above yearly sales graphs, we can conclude that:
# * The year 2017 has seen a major dip in the sales of drugs. This need digging. Lets do it

# In[ ]:


#lets see some statistics related to the data
monthly.describe()


# Here, we see that the minimum value of sale of majority of drugs is 0 while that of drug N05B is 1. This is the reason why year 2017 has lowest sales.

# In[ ]:


#plot line curve to analyse monthly sales
def plot_line_curve(series):
    plt.figure(figsize= (15,5))
    series.plot(kind= 'line')
    plt.title(f'Monthly Sales of Drug : {col}')
    plt.show()


# In[ ]:


for col in monthly.columns[1:9]:
    plot_line_curve(monthly[col])


# From the above graphs, we can infer that the sales for first month of 2017 is 0. This means that we have missing values for the first month.
# Let us analyse this from daily data.<br>
# But first let us preprocess daily data also.

# In[ ]:


daily.columns


# In[ ]:


#extracting days from date
daily['day'] = daily['datum'].dt.day


# In[ ]:


#set dates as index
daily.set_index(daily['datum'], inplace= True)


# In[ ]:


#looking at sales data from 1st Jan, 2017 to 1st Feb, 2019
for col in daily.columns[1:9]:
    plot_line_curve(daily[col].loc['1/1/2017':'2/1/2017'])


# From these graphs, we can say say that the data is **not missing**. Instead, the sales of drugs on 2nd January, 2017 is low rather there is no sale on 2nd Feb.

# ## Analysing total sales of drug

# In[ ]:


#calculating total sales
monthly['total_sales'] = monthly['M01AB']
for cols in monthly.columns[2:9]:
    monthly['total_sales'] = monthly['total_sales']+monthly[cols]


# In[ ]:


monthly.groupby('month')['total_sales'].plot.bar(rot=45)
plt.xlabel('Date Time')
plt.ylabel('Total Sales')
plt.title('Total Sales of Drugs')
plt.show()


# From above diagram we can hence validate that the sales of drugs have been lowest in the year 2017.
