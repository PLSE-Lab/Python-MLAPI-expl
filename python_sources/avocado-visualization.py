#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/avocado-prices/avocado.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()
#there is no nan in data


# In[ ]:





# In[ ]:


df.drop(columns='year',inplace=True)


# In[ ]:


# dealing with 'Date' column and splitting it to 'day' , 'Month' and year columns

df[['Year', 'Month','day']]=df.Date.str.split('-', expand = True)


# In[ ]:


df.head()


# In[ ]:


df.drop(columns='Date',inplace=True)


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.region.unique()


# In[ ]:


df.tail(20)


# In[ ]:


df.region.value_counts()


# In[ ]:


sns.barplot(x='type',y='AveragePrice',data=df)


# In[ ]:


sns.barplot(x='type',y='Total Volume',data=df)


# In[ ]:


sns.countplot(x='type',data=df)


# In[ ]:


sns.distplot(df['AveragePrice'])


# In[ ]:


sns.distplot(df['Total Volume'])


# In[ ]:


sns.boxplot(x="Year", y="AveragePrice", data=df)


# In[ ]:


sns.boxplot(x="Month", y="AveragePrice", data=df)


# In[ ]:


sns.scatterplot(x="Month", y="AveragePrice", data=df)


# In[ ]:


sns.barplot(x="Month", y="AveragePrice",hue='type', data=df)
#price of orcanic more than conventional


# In[ ]:


sns.barplot(x="Month", y="Total Volume",hue='type', data=df)
# conventional sold more than orcanic because the brice is little than orcanic


# In[ ]:


df.head()


# In[ ]:


df['Unnamed: 0'].value_counts()


# In[ ]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


df.describe()


# In[ ]:


sns.heatmap(df.corr(),cmap='magma',linecolor='white',linewidths=1)


# # find the maxmum valuse by catigorical columns

# In[ ]:


AVG_REG = df[['region', 'AveragePrice']].groupby('region').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()


# In[ ]:


AVG_REG.head()


# In[ ]:


print('The country with the lowest AveragePrice is ','("',(AVG_REG.min()[0]),'")','with value','("',(AVG_REG.min()[1]),'")')
print('The country with the highest AveragePrice is ','("',(AVG_REG.max()[0]),'")','with value','("',(AVG_REG.max()[1]),'")')


# In[ ]:


total_REG = df[['region', 'Total Volume']].groupby('region').agg('mean').sort_values(by = 'Total Volume', ascending = False).reset_index()


# In[ ]:


total_REG.head()


# In[ ]:


print('The country with the lowest Total Volume is ','("',(total_REG.min()[0]),'")','with value','("',(total_REG.min()[1]),'")')
print('The country with the highest Total Volume is ','("',(total_REG.max()[0]),'")','with value','("',(total_REG.max()[1]),'")')


# In[ ]:


total_type = df[['type', 'Total Volume']].groupby('type').agg('mean').sort_values(by = 'Total Volume', ascending = False).reset_index()


# In[ ]:


total_type


# In[ ]:


ava_type = df[['type', 'AveragePrice']].groupby('type').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()


# In[ ]:


ava_type


# # find max and min valuse with date columns

# In[ ]:


ava_day = df[['day', 'AveragePrice']].groupby('day').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()


# In[ ]:


ava_day.head()
#The most 5 days the AveragePrice is high in them


# In[ ]:


ava_month = df[['Month', 'AveragePrice']].groupby('Month').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()


# In[ ]:


ava_month.head()
#The most 5 months the AveragePrice is high in them


# In[ ]:


ava_year = df[['Year', 'AveragePrice']].groupby('Year').agg('mean').sort_values(by = 'AveragePrice', ascending = False).reset_index()


# In[ ]:


ava_year


# In[ ]:


total_year = df[['Year', 'Total Volume']].groupby('Year').agg('mean').sort_values(by = 'Total Volume', ascending = False).reset_index()


# In[ ]:


total_year


# # visualizing data
# 

# In[ ]:


df.head()


# In[ ]:


sns.clustermap(df.corr(),cmap='coolwarm',annot=True)


# In[ ]:


sns.lineplot(x='AveragePrice',y='Total Volume',data=df)


# In[ ]:


sns.scatterplot(x='AveragePrice',y='Total Volume',data=df)


# In[ ]:


sns.barplot(x='Month',y='Total Volume',hue='type',data=df,estimator=np.std)


# In[ ]:


sns.barplot(x='Month',y='Total Bags',hue='type',data=df)


# In[ ]:


sns.countplot(df.type)


# In[ ]:


sns.lineplot(x="Month",y='AveragePrice',data=df)
#high price in octoper 


# In[ ]:


sns.lineplot(x="Month",y='Total Volume',data=df)
# 


# In[ ]:


#sns.lineplot(x='4046',y='Total Volume',data=df)


# In[ ]:


#sns.lineplot(x='Total Volume',y='4770',data=df)
#ther is no 


# In[ ]:


sns.boxplot(x="Year", y="AveragePrice", data=df,palette='rainbow')


# In[ ]:


sns.boxplot(x="Month", y="AveragePrice", data=df,palette='rainbow')


# In[ ]:


sns.boxplot(x="day", y="AveragePrice", data=df,palette='rainbow')


# In[ ]:


plt.figure(figsize=(12,7))
sns.lineplot(x='day',y='AveragePrice',data=df)


# In[ ]:


#sns.pairplot(df,hue='type',palette='rainbow')


# In[ ]:


g = sns.PairGrid(df,hue='type')
g.map(plt.scatter)
# orange for orcanic blue for athor


# In[ ]:


df.head()


# In[ ]:


#g = sns.JointGrid(x="Total Volume", y="AveragePrice", data=df)
#g = g.plot(sns.regplot, sns.distplot)


# In[ ]:


sns.lmplot(x='Total Volume',y='4225',data=df,col='type')


# In[ ]:


#g = sns.JointGrid(x="Total Volume", y="Total Bags", data=df)
#g = g.plot(sns.regplot, sns.distplot)
sns.lmplot(x='Total Volume',y='Total Bags',data=df,col='type')


# In[ ]:


df.head()


# In[ ]:


g = sns.FacetGrid(df, col = 'type', row = 'Year', palette = 'RdBu_r',hue = 'region',  height = 3.5, aspect = 2)
g.map(sns.scatterplot, 'Total Volume', 'Total Bags')
g.add_legend()
plt.show()


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),cmap='coolwarm',linecolor='white',annot=True,linewidths=1)


# # This is a short story that tells us some information about this data
# 
After analyzing this data, we found that there are two types of avocados, one of which is organic and the other conventional. The same amount has been sold, but with many differences.
The average price of organic was higher, but Total Volume was significantly higher in conventional than organic
The average price ranged from 0.5 to 2.5, where 2017 was the year in which the average price was higher than usual, while 2018 was the year in which the maximum value of the average price decreased
As for the months, the month 9 was the owner of the largest price
There was a strong relationship between all the data, so the lower the average price, the more the rest of all the elements decreased.
For Large Bags and Small Bags, the less the relative proportions of organic


The country with the lowest Total Volume is ("Albany") with value ("32374.756449704146")
The country with the highest Total Total is ("WestTexNewMexico") with value ("17351302.31301776")


The country with the lowest AveragePrice is ("Albany") with value ("1.0479289940828398")
The country with the highest AveragePrice is ("WestTexNewMexico") with value ("1.8186390532544363")