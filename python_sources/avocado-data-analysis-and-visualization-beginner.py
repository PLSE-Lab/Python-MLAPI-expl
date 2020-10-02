#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/avocado-prices/avocado.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


#missing values
data.isnull().sum()

A clean Dataset from missing values 
# In[ ]:


## i will rename some columns for more flexibility
data.rename(columns={'AveragePrice':'avprice','Small Bags':'Sbags','Large Bags':'Lbags','XLarge Bags':'XLbags'},inplace=True)


# In[ ]:


data.head()


# ## AveragePrice analysis

# In[ ]:


data.avprice.describe()


# In[ ]:


sns.set(style='darkgrid')
plt.figure(figsize=(11,9))
a=sns.distplot(data.avprice,color='r')
a.set_xlabel('AvragePrice')
a.set_ylabel('Frequency')
plt.title('Distribution of Average Price',size=25)


# ## Totale Volume

# In[ ]:


data['Total Volume'].describe()


# In[ ]:


plt.figure(figsize=(11,9))
a=sns.kdeplot(data['Total Volume'],color='g',shade=True)
a.set_xlabel('Total Volume')
a.set_ylabel('Frequency')
plt.title('Distribution of Total Volume',size=25)


# In[ ]:


data[data['Total Volume']<1500000].shape


# In[ ]:


data[data['Total Volume']>5000000].sort_values(by='Total Volume',ascending=False)

Most of totale volume are less than 1,5M
# In[ ]:


## correlation between them 
print('the correlation between AveragePrice and Total volume :',data['avprice'].corr(data['Total Volume']))


# In[ ]:


a=sns.jointplot(x='Total Volume',y='avprice',data=data,color='g',height=9)


# In[ ]:


plt.figure(figsize=(11,9))
a=sns.regplot(x='Total Volume',y='avprice',data=data[data['Total Volume']<1500000],color='c')
plt.title('Average Price vs Total Volume',size=25)


# ## Results :
It's clear that total volume affect to Average Price which is normal,
as long as the volume increase, the price goes down
# ## Region

# In[ ]:


data.region.unique()


# In[ ]:


Region=data.groupby('region').avprice.mean().reset_index().sort_values(by='avprice')


# In[ ]:


Region.head()


# In[ ]:


Region.tail()


# In[ ]:


plt.figure(figsize=(11,9))
a=sns.boxplot(x='region',y='avprice',data=data,palette='nipy_spectral')
a.set_xticklabels(a.get_xticklabels(), rotation=90, ha="right",size=12)
plt.title('Boxplot AveragePrice Vs Region',size=30)


# ## Results :
 Houston,DallasFtWorth and SouthCentral have the lowest average price 
# ## Type

# In[ ]:


data.type.unique()


# In[ ]:


data.type.value_counts()


# In[ ]:


data.groupby('type').avprice.mean().reset_index()


# In[ ]:


a=sns.catplot(x='type',y='avprice',data=data,palette='mako',height=10,kind='boxen')
plt.title('Boxen plot of AverigePrive for each type',size=25)


# In[ ]:


a=sns.catplot(x='type',y='Total Volume',data=data[data['Total Volume']<1500000],palette='mako',height=10,kind='box')
plt.title('Boxen plot of Total Volume for each type',size=25)


# In[ ]:


sns.relplot(x="Total Volume",y='avprice',hue='type',data=data,height=10)
plt.title('AveragePrice Vs Total Volume for each type',size=25)


# In[ ]:


plt.figure(figsize=(13,15))
a=sns.barplot(x='avprice',y='region',data=data,palette='nipy_spectral',hue='type')
a.set_yticklabels(a.get_yticklabels(),size=16)
plt.title('Barplot AveragePrice Vs Region for each Type',size=30)


# In[ ]:


plt.figure(figsize=(10,15))
a=sns.barplot(x='Total Volume',y='region',data=data[data.type=='organic'].query('region != "TotalUS"'),palette='coolwarm')
a.set_yticklabels(a.get_yticklabels(),size=16)
plt.title('Total Volume for organic for each Region',size=30)


# In[ ]:


plt.figure(figsize=(10,15))
a=sns.barplot(x='Total Volume',y='region',data=data[data.type=='conventional'].query('region != "TotalUS"'),palette='coolwarm')
a.set_yticklabels(a.get_yticklabels(),size=16)
plt.title('Total Volume for conventional for each Region',size=30)


# ## Results :
The organic avocado it's more expensive than conventional, because it is not produced in large quantities
generally speaking, organic food suply is limitd as compared to demand and production costs for organic foods are typically higher.

# ## Year

# In[ ]:


data.year.unique()


# In[ ]:


data.Date=pd.to_datetime(data.Date)


# In[ ]:


data['month']=data.Date.dt.month


# In[ ]:


price_years=data.groupby(['year','month','type'],as_index=False)['avprice'].mean()
price_years


# ## Let's see the mean of AveragePrice each month over  years

# ##  2015

# In[ ]:


plt.figure(figsize=(13,9))
a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2015"')
            ,hue='type',markers=True,style='type'
            ,palette='gnuplot2' )


# ## 2016

# In[ ]:


plt.figure(figsize=(13,9))
a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2016"')
            ,hue='type',markers=True,style='type'
            ,palette='gnuplot2' )


# # 2017

# In[ ]:


plt.figure(figsize=(13,9))
a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2017"')
            ,hue='type',markers=True,style='type'
            ,palette='gnuplot2' )


# ## 2018

# In[ ]:


plt.figure(figsize=(13,9))
a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2018"')
            ,hue='type',markers=True,style='type'
            ,palette='gnuplot2' )


# ## Results :
-In 2015, the average price of conventional avocado was in kind of stabilization  and did not exceed 1.15 over all months .
-The year 2017 had the highest  average price of avocado exactly on september .
# ## Averge Pice of Avocado for each Region over the four years

# In[ ]:


sns.factorplot('avprice','region',data=data.query("type=='conventional'"),
                hue='year',
                size=15,
                palette='tab20',
                join=False,
                aspect=0.7,
              )
plt.title('For Conventional',size=25)


# In[ ]:


sns.factorplot('avprice','region',data=data.query("type=='organic'"),
                hue='year',
                size=15,
                palette='tab20',
                join=False,
                aspect=0.7,
              )
plt.title('For Organic',size=25)


# ## Results :
the organic avocado was expensive in 2017 especially on sanFrancisco and hartford-springfield
# ### !!! Thank you waiting for your remarks

# In[ ]:




