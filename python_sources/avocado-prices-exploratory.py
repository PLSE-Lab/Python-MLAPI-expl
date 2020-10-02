#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/avocado.csv')
df.head()


# Show the shape of our data (columns ,rows)

# In[ ]:


df.shape


# Check missing data

# In[ ]:


df.isnull().sum()


# Good there is not missing data.
# 

# In[ ]:


df.info()


# Remove the fist colum . no need for it .

# In[ ]:


df=df.drop(['Unnamed: 0'],axis=1)


# In[ ]:


df.info()


# Convert the type of 'Data' column to datetime type

# In[ ]:


df['Date']=pd.to_datetime(df['Date'])


# Remove sapce in columns names

# In[ ]:



df.columns=df.columns.str.replace(" ","_")


# In[ ]:


df.columns


# Now, let's investigate the data in each column.

# Display how many regions there are , and what are.

# In[ ]:


region=df['region'].unique().tolist()
len(region)


# In[ ]:


print(region)


# Display the years

# In[ ]:


years=df['year'].unique().tolist()
print(years)


# investigate Type column

# In[ ]:


df['type'].value_counts()


# There are tow type of avocado: conventional and organic.

# Sorting data by date

# In[ ]:


df=df.sort_values(by=('Date'))


# In[ ]:


plt.figure(figsize=(22,12))
plt.title('Average price over Years')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.plot(df['Date'].values,df["AveragePrice"].rolling(88).mean().values)
plt.show()


# In[ ]:


df_organic=df[df['type'].str.contains('organic')]
df_conventional=df[df['type'].str.contains('conventional')]


# In[ ]:





# In[ ]:


plt.figure(figsize=(20,10))
plt.xlabel('Average Price')
plt.ylabel('Total')
plt.xlim(0,3.5)
plt.ylim(0,3000)
plt.title('Distribution Price')
plt.grid(True)
plt.hist(df['AveragePrice'],bins=30)
plt.show()


# In[ ]:


figure,axs=plt.subplots(1,3,figsize=(25,6))

axs[0].set(xlabel='Average Price')
axs[0].set(ylabel='Total')
axs[0].set(xlim=(0,3.5))
axs[0].set(ylim=(0,3000))
axs[0].set_title('Distribution Price of All Types')
axs[0].hist(df['AveragePrice'],bins=30)

axs[1].set(xlabel='Average Price of Organic Type')
axs[1].set(ylabel='Total')
axs[1].set(xlim=(0,3.5))
axs[1].set(ylim=(0,3000))
axs[1].set_title('Distribution Price of Organic Type')
axs[1].hist(df_organic['AveragePrice'],bins=30)

axs[2].set(xlabel='Average Price of Organic Type')
axs[2].set(ylabel='Total')
axs[2].set(xlim=(0,3.5))
axs[2].set(ylim=(0,3000))
axs[2].set_title('Distribution Price of conventional Type')
axs[2].hist(df_conventional['AveragePrice'],bins=33)



# In[ ]:


print ("The hisogram show the average of avocado priceof all types is  ",df['AveragePrice'].mean())
print ("The hisogram show the average of Organic avocado price is  ",df_organic['AveragePrice'].mean())
print ("The hisogram show the average of Conventional Avocado price is  ",df_conventional['AveragePrice'].mean())


# In[ ]:


plt.figure(figsize=(22,5))
plt.title('Average Price over Years')
plt.xlabel('Date ')
plt.ylabel('Average Price')
plt.plot(df['Date'].values,df["AveragePrice"].rolling(88).mean().values,label='All type')
plt.plot(df_organic['Date'].values,df_organic["AveragePrice"].rolling(88).mean().values, label='Organic')
plt.plot(df_conventional['Date'].values,df_conventional["AveragePrice"].rolling(88).mean().values, label='Conventional')
plt.legend(loc='best')
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.title('Total Volume over Years')
plt.xlabel('Date ')
plt.ylabel('Total Volume')
plt.plot(df['Date'].values,df['Total_Volume'].rolling(100).mean().values,label='All Type')
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.title('Total Volume over Years')
plt.xlabel('Date ')
plt.ylabel('Total Volume')
plt.plot(df_conventional['Date'].values,df_conventional['Total_Volume'].rolling(80).mean().values,label='Conventional')
plt.plot(df_organic['Date'].values,df_organic['Total_Volume'].rolling(80).mean().values,label='organic')
plt.legend(loc='best')
plt.show()


# Above figure shows that total volume of conventional Avocado is larger than total volume of organic Avocado over time.
# Also, total volume of organic Avocado is stable over time , but for conventional avocado its total volues is increasing over time im general ,although it dereased in several period 

# In[ ]:


plt.figure(figsize=(20,9))
plt.title('Average Price VS Total Volume  ')
plt.xlabel('Average Price ')
plt.ylabel('Total Volume')
plt.scatter(df['AveragePrice'],df['Total_Volume'])
plt.show()


# The figure shows there is an increase in sales if sold at its average price, but it is clear that when price increases , the total volume of sold avocado decrease. 
# Correlation between average price and total volume we could compute it as :

# In[ ]:


print(df['AveragePrice'].corr(df['Total_Volume']))


# In[ ]:


plt.figure(figsize=(20,10))
plt.boxplot([df['AveragePrice'].values,df_organic['AveragePrice'].values,df_conventional['AveragePrice'].values],labels=['All Type','organic','conventional'])

plt.show()


# In[ ]:


df_conventional_2015=df_conventional[df_conventional['year']<2016]
df_conventional_2016=df_conventional[(df_conventional['year']>2015) & (df_conventional['year']<2017)]
df_conventional_2017=df_conventional[(df_conventional['year']>2016) & (df_conventional['year']<2018)]
df_conventional_2018=df_conventional[(df_conventional['year']>2017) & (df_conventional['year']<2019)]

df_organic_2015=df_organic[df_organic['year']<2016]
df_organic_2016=df_organic[(df_organic['year']>2015) & (df_organic['year']<2017)]
df_organic_2017=df_organic[(df_organic['year']>2016) & (df_organic['year']<2018)]
df_organic_2018=df_organic[(df_organic['year']>2017) & (df_organic['year']<2019)]


# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Total Volume of Conventional Type over Years')
plt.xlabel('Date ')
plt.ylabel('Total Volume of Conventional Type')
plt.plot(df_conventional_2015['Date'],df_conventional_2015['Total_Volume'].rolling(122).mean().values, label='2015')
plt.plot(df_conventional_2016['Date'],df_conventional_2016['Total_Volume'].rolling(122).mean().values, label='2016')
plt.plot(df_conventional_2017['Date'],df_conventional_2017['Total_Volume'].rolling(122).mean().values, label='2017')
plt.plot(df_conventional_2018['Date'],df_conventional_2018['Total_Volume'].rolling(122).mean().values, label='2018')
plt.legend(loc='best')
plt.show()


# Total volume of conventional type is increasing over years.

# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Total Volume of df_organic Type over Years')
plt.xlabel('Date ')
plt.ylabel('Total Volume of df_organic Type')
plt.plot(df_organic_2015['Date'],df_organic_2015['Total_Volume'].rolling(122).mean().values, label='2015')
plt.plot(df_organic_2016['Date'],df_organic_2016['Total_Volume'].rolling(122).mean().values, label='2016')
plt.plot(df_organic_2017['Date'],df_organic_2017['Total_Volume'].rolling(122).mean().values, label='2017')
plt.plot(df_organic_2018['Date'],df_organic_2018['Total_Volume'].rolling(122).mean().values, label='2018')
plt.legend(loc='best')
plt.show()


# Total volume of organic type is increasing over years.

# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Average Price  of Conventional Type over Years')
plt.xlabel('Date ')
plt.ylabel('Average Price  of Conventional Type')
plt.plot(df_conventional_2015['Date'],df_conventional_2015['AveragePrice'].rolling(55).mean().values,label='2015')
plt.plot(df_conventional_2016['Date'],df_conventional_2016['AveragePrice'].rolling(55).mean().values,label='2016')
plt.plot(df_conventional_2017['Date'],df_conventional_2017['AveragePrice'].rolling(55).mean().values,label='2017')
plt.plot(df_conventional_2018['Date'],df_conventional_2018['AveragePrice'].rolling(55).mean().values,label='2018')
plt.legend(loc='best')
plt.show()


# Average price of conventional type is icreasing over years, and most increasing was in 2017.

# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Average Price  of Organic Type over Years')
plt.xlabel('Date ')
plt.ylabel('Average Price  of Organic Type')
plt.plot(df_organic_2015['Date'],df_organic_2015['AveragePrice'].rolling(55).mean().values,label='2015')
plt.plot(df_organic_2016['Date'],df_organic_2016['AveragePrice'].rolling(55).mean().values,label='2016')
plt.plot(df_organic_2017['Date'],df_organic_2017['AveragePrice'].rolling(55).mean().values,label='2017')
plt.plot(df_organic_2018['Date'],df_organic_2018['AveragePrice'].rolling(55).mean().values,label='2018')
plt.legend(loc='best')
plt.show()


# Average price of Organic type is icreasing over years

# In[ ]:


region_AveragePrice=df[['region','AveragePrice']].groupby('region').mean().sort_values(by='AveragePrice')
print(region_AveragePrice)


# The lowest Average price in Houston( 1.05) , and the highest average price in HartfordSpringfield (1.82).

# In[ ]:


region_tot_volume=df[['region','Total_Volume']].groupby('region').mean().sort_values(by='Total_Volume')
print(region_tot_volume)


# The lowest total volume of sold Avocado in Syracuse , and the highest total volume in TotalUS.

# In[ ]:




