#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
df = pd.DataFrame(data)


# In[ ]:


#print(df.head(10))
print(df.tail(10))


# In[ ]:


print(df.shape)


# In[ ]:


print(df.info())


# In[ ]:


print(df.columns)


# In[ ]:


print(df['Country'].unique())


# In[ ]:


df['Last Update'] = pd.to_datetime(df['Last Update'])

df['date'] = df['Last Update'].dt.date
df['Time'] = df['Last Update'].dt.time
print(df[['date','Time']])


# In[ ]:


df_date = df.groupby('date')['Confirmed'].sum().reset_index()
df_date.columns = ['Date','Affected']

df_date['Affect_everyday'] = df_date['Affected'] - df_date['Affected'].shift(1)
df_date.columns = ['Date','Affected','Affect_everyday']
df_date['Affect_everyday'] = df_date['Affect_everyday'].fillna('555.0')
print(df_date)   


# Remember we are seeing data after 21 Jan, so all people affected before that is added to 21 Jan. 
# Now we are going to see all the people affected till that day, remember everyday adds new and previous affected. So Total affected people is only on last date.
# For people affected on each day, I made a column (Affect_everyday) above.

# In[ ]:


plt.figure(figsize = (16,9))
sns.barplot('Date','Affected', data = df_date)
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Numbers of Infected', fontsize = 16)
plt.title('Total Numbers of Infected every Passing Day', fontsize = 20)
plt.show()


# In[ ]:


#df_country = df.groupby('Country')['Confirmed'].count().reset_index().sort_values(by='Confirmed',ascending = False)[0:15]
df_country = df.groupby('Country')['Confirmed'].max().reset_index().sort_values('Confirmed', ascending = False)[0:15]
df_country.columns = ['Country','Affected']
print(df_country)


plt.figure(figsize = (16,9))
sns.barplot(df_country['Country'], df_country['Affected'])
plt.xticks(rotation = 70)
plt.title('Bar Chart of Numbers of people affected in each country WITH CHINA')
plt.xlabel('Countries that are affected by Coronavirus')
plt.ylabel('Number of people affected in each country')
plt.show()


# Above Graphs clearly shows that maximum number of people affected by coronavirus is in China which is the origin of virus, which single handly undermining the number of people affected in other country.
# Now, we will see number of people affected in different countries except China.

# In[ ]:


df_country1 = df.groupby('Country')['Confirmed'].max().reset_index().sort_values('Confirmed', ascending = False)[2:24]
df_country1.columns = ['Country','Affected']
plt.figure(figsize = (16,9))
sns.barplot(df_country1['Country'], df_country1['Affected'])
plt.xticks(rotation = 70)
plt.title('Bar Chart of Numbers of people affected in each country WITHOUT CHINA')
plt.xlabel('Countries that are affected by Coronavirus')
plt.ylabel('Number of people affected in each country')
plt.show()


# In[ ]:


df_country_ec = df_country[0:24]
print(df_country_ec)


# Cities affected by Virus

# In[ ]:


df_city = df.groupby('Province/State')['Confirmed'].max().reset_index()
print(df_city)


# In[ ]:


df_city = df.groupby('Province/State')['Confirmed'].max().reset_index().sort_values('Confirmed', ascending = False)[0:20]
print(df_city)

plt.figure(figsize = (16,9))
sns.barplot(df_city['Province/State'], df_city['Confirmed'])
plt.xticks(rotation = 70)
plt.title('Bar Chart of Numbers of people affected in each region')
plt.xlabel('Chinese Region that are affected by Coronavirus')
plt.ylabel('Number of people affected in each country')
plt.show()


# You can clearly see above Wuhan which is the part of Hubei province saw the maximum number of People affected by Coronavirus.
