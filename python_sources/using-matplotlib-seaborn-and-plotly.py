#!/usr/bin/env python
# coding: utf-8

# # We will Start by importing necessary Libraries

# In[ ]:


#Data analysis and wrangling
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected = True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Acquiring the data
# The Python Pandas packages helps us work with our datasets. We start by acquiring the dataset into Pandas DataFrames.

# In[ ]:


#Reading our csv file into pandas dataframe
df = pd.read_csv('../input/train (3).csv')
df.head()


# # Now we will try to understand our data before doing any analysis
# 

# In[ ]:


#Viewing the shape of the data to know how many features we have and how many rows 
df.shape


# In[ ]:


#As we can see there are null values on the Province_State
df.info()


# In[ ]:


#How many null values do we have in each feature 
df.isnull().sum()


# In[ ]:


#Although, we won't care much about dealing with missing data for now
#But Plotting missing data can be very helpful in understanding our dataset
total = df.isnull().sum().sort_values(ascending=False)
percent = ((df.isnull().sum()/df.isnull().count()) * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()


# # Now that we have a little bit understanding about our data
# ## Let's now determine which features are important in our analysis and which is not

# In[ ]:


# How many unique values in our Id column
# I know this sounds strange as when seeing id you know that it will be unique but some datasets have duplicated id values 
df.Id.nunique()


# In[ ]:


# Now we are trying to find how many unique values are presented in Province_state column 
# Finding unique values may help us have better understanding of our data
print(f"It has { df.Province_State.nunique() } unique values and it's top 5 values are:")
print('-'*50)
print(df.Province_State.value_counts(dropna=False).head())


# In[ ]:


# trying to find how many unique values are presented in Country_Region column 
print(f"It has { df.Country_Region.nunique() } unique values and it's top 5 values are:")
print('-'*50)
print(df.Country_Region.value_counts(dropna=False).head())


# In[ ]:


df.ConfirmedCases.nunique()


# In[ ]:


print(df.columns.values)


# # Now as you can see that none of those columns can give us better information 
# ## But we have the Date column which may help us understand more about our data
# we must make sure that it's a datetime format cause as we saw above it is an object format  

# In[ ]:


# Convert Date from object to datetime64 format
df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


# As we can see day has 31 unique values which is better than before 
df.Date.dt.day.nunique()


# In[ ]:


# Month is even better as it has only 4 values
df.Date.dt.month.nunique()


# In[ ]:


# Year won't help us.. 
df.Date.dt.year.nunique()


# ## We can see now that date column actually can help us very much in our analysis 

# In[ ]:


# Create two columns for the day and month 
df['Day'] = df.Date.dt.day
df['Month'] = df.Date.dt.month


# In[ ]:


df.head()


# In[ ]:


# We can drop the Date column as it we got what we wanted from it 
# we can also drop the id column as it doesn't give us any infromation 
df.drop(['Id', 'Date'], axis=1, inplace=True)


# In[ ]:


# Our dataframe now looks like this
df.head()


# In[ ]:


# Our features data types
df.dtypes


# In[ ]:


# Now we want to know how to use the month column in our advantage
df.Month.value_counts(dropna=False)


# In[ ]:


# Use describe function to know more about the month and if we can use it or not 
df.groupby('Month').describe()


# # Time for some visualization using matplotlib & seaborn

# In[ ]:


# Create a frequency distribution table as our month is a categorical variable although it's integer 
month_freq = df.groupby('Month').size().reset_index(name='Count')
plt.figure(figsize=(15, 10))
sns.set_style('whitegrid')
sns.countplot(x='Month', data=df)
plt.xlabel('Month', fontsize=20)
plt.ylabel('Count', fontsize=20)
month_freq


# In[ ]:


# Pie chart
# values for the chart
val = [df['Month'][df['Month'] == 1].count(),df['Month'][df['Month'] == 2].count(), df['Month'][df['Month'] == 2].count()]  # number of values of Jan, Feb & March
fig1, ax1 = plt.subplots(figsize=(15, 7))
ax1.pie(val, explode=(0, 0.05, 0.05), labels=['January', 'February', 'March'], colors=['#c03d3e','#095b97', '#3a923a'], autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 15, 'fontWeight':'bold'})
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()


# In[ ]:


# Using histogram for Fatalities 
plt.figure(figsize=(18, 10))
plt.hist(df.Fatalities)


# In[ ]:


# Using the distribution plot from seaborn 
plt.figure(figsize=(18, 10))
sns.distplot(df.Fatalities.dropna(), bins=30)


# In[ ]:


# Creating a scatter plot for ConfirmedCases & Fatalities
plt.figure(figsize=(15, 10))
plt.scatter(df.ConfirmedCases, df.Fatalities, marker='D')


# In[ ]:


sns.lmplot(x='ConfirmedCases', y='Fatalities', data=df, hue='Month', markers=['o', 'D', 'D', '*'])


# In[ ]:


# stripplot showed that every month number of cases and Fatalities increases
plt.figure(figsize=(15, 7))
sns.stripplot(x='Month', y='Fatalities', data=df)


# In[ ]:


sns.catplot(x='Month', y='Fatalities', data=df, height=10)


# In[ ]:


sns.catplot(x='Month', y='ConfirmedCases', data=df, height=10)


# In[ ]:


sns.pairplot(df, hue='Month', diag_kind='hist')


# In[ ]:


# Creating a coorelation heatmap to how features are affected by each other (relationship between features)
# using correlation coefficient 
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')


# # Using Plotly

# In[ ]:


# Creating a barplot for month column
df.iplot(kind = 'bar', x = 'Month', xTitle='Month', yTitle='Count')


# In[ ]:


# Seeing the relation between Month and Fatalities
df.iplot(kind = 'bar', x = 'Month', y = 'Fatalities', xTitle='Month', yTitle='Fatalities')


# In[ ]:


# Using a Scatter plot for the confirmedCases and fatalities   
df.iplot(kind = 'scatter', x = 'ConfirmedCases', y = 'Fatalities', mode='markers',symbol='circle-dot',colors=['orange','teal'], size=20)


# In[ ]:


df[['ConfirmedCases', 'Fatalities']].iplot(kind='spread')


# In[ ]:


df.iplot(x = 'ConfirmedCases', y = 'Fatalities')


# In[ ]:


df['Fatalities'].iplot(kind = 'hist', bins = 25)


# In[ ]:


# Box plot for our dataset 
df.iplot(kind = 'box')


# In[ ]:


df.iplot(kind = 'bubble', x = 'Fatalities', y = 'ConfirmedCases', size = 'Day')


# In[ ]:


df.iplot()


# In[ ]:


df[['ConfirmedCases', 'Fatalities']].iplot(kind='area',fill=True,opacity=1)


# In[ ]:


df.Country_Region.iplot()


# In[ ]:




