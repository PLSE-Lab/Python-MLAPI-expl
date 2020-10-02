#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/uberdrives/My Uber Drives - 2016.csv")


# In[ ]:


df.isnull().sum()


# The above values shows there are one missing value in few columns which we can drop for our analysis.Inorder to find out which row has the missing values,we should do a mask.

# In[ ]:


df[df['END_DATE*'].isnull()]


# In[ ]:


df.drop(df.index[1155],inplace=True)


# After finding which row has the missing values we are dropping that row by the index value

# In[ ]:


df.isnull().sum()


# Checking the data types for the variables

# In[ ]:


df.head()


# In[ ]:


df.dtypes


# Since the date and time of the start_date and end_date are of data type object we are converting it to datetime

# In[ ]:


df['START_DATE*'] = df['START_DATE*'].astype('datetime64[ns]')
df['END_DATE*'] = df['END_DATE*'].astype('datetime64[ns]')


# ### From the below chart we can infer that cabs are more used for business purpose than for personal use and the count of travels for each category

# In[ ]:


a=pd.crosstab(index=df['CATEGORY*'],columns='Count of travels as per category')
a.plot(kind='bar',color='r',alpha=0.7)
plt.legend()
a


# ### From the below plot we can infer that most of the cabs are taken for meeting purpose

# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(df['PURPOSE*'],order=df['PURPOSE*'].value_counts().index)


# The below graph shows the mean distance travelled per ride for business and personal purposes

# In[ ]:


df.groupby('CATEGORY*')["MILES*"].mean().plot(kind='bar',color='g')
plt.axhline(df["MILES*"].mean(),label='Mean distance travelled per ride')
plt.legend()


# Below we are creating a variable as Round_trip if the start and stop place are same

# In[ ]:


df['Round_trip'] = df.apply(lambda x : 'Yes' if x['START*'] == x["STOP*"] else 'no',axis=1)


# In[ ]:


coun=pd.crosstab(df['Round_trip'],df['CATEGORY*'])
per=coun.div(coun.sum(1),axis=0)*100
per.plot(kind='bar',stacked=True)
plt.legend(bbox_to_anchor=(1.05,1),loc=2)
round(per,2)


# ### The above graph shows the Percentage of roundtrips for business and personal purposes. And we can also infer that one way trips are More than round trips
# 
# 
# 
# 
# ### Since the months are of integer we are converting it to month name

# In[ ]:


df['Month'] = pd.DatetimeIndex(df['END_DATE*']).month
df['Month']


# In[ ]:


s= {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df['Month'] = df['Month'].map(s)


# In[ ]:


df['Month'].dtypes


# In[ ]:


a=sns.countplot(df['Round_trip'],hue=df['Month'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)


# ### From the above plot we can infer that there is a high amount of round trips in the month of December

# In[ ]:


c=pd.crosstab(index=df['Month'],columns='Number of trips')
c.sort_values('Number of trips',ascending=False).plot(kind='bar',color='y')
plt.axhline(c['Number of trips'].mean(),linestyle='--')


# From the above plot we can see that most rides are in the month of december and least in the month of november

# In[ ]:


sns.countplot(df['Month'],hue=df['CATEGORY*'],order=df['Month'].value_counts().index)


# From the above graph we can infer that takes are taken for personal use only in the months of feb,march,april,jun,july

# In[ ]:


df


# ### We are creating a new variable as day or night ride if the start time is more than 6 pm its night ride else day ride

# In[ ]:


df['Day/Nightride'] = pd.DatetimeIndex(df['START_DATE*']).time


# In[ ]:


a = pd.to_datetime(['18:00:00']).time


# In[ ]:


df['Day/Nightride'] = df.apply(lambda x : 'Night ride' if x['Day/Nightride'] > a else 'Day ride',axis=1)


# In[ ]:


sns.countplot(df['Day/Nightride'],hue=df['CATEGORY*'])


# ### From the above plot we can see that across both business and personal purpose night rides are minimum

# ### Converting the continous variable miles into buckets

# In[ ]:


f = {}
for i in df['MILES*']:
    if i < 10:
        f.setdefault(i,'0-10 miles')
    elif i >= 10 and i < 20:
        f.setdefault(i,'10-20 miles')
    elif i >= 20 and i < 30:
        f.setdefault(i,'20-30 miles')
    elif i >= 30 and i < 40:
        f.setdefault(i,'30-40 miles')
    elif i >= 40 and i < 50:
        f.setdefault(i,'40-50 miles')
    else:
        f.setdefault(i,'Above 50 miles')


# In[ ]:


df['MILES*'] = df['MILES*'].map(f)


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(df['MILES*'],order=df['MILES*'].value_counts().index)


# ### From the above plot we can see that cabs are more often used for short distances

# In[ ]:


f = pd.crosstab(df['Month'],df["MILES*"])
f.plot(kind='bar')
plt.legend(bbox_to_anchor=(1.05,1),loc=2)
f


# from the above data we can see the category of miles driven for each month

# In[ ]:


z = df.groupby('Month')['Day/Nightride'].count().mean()


# In[ ]:


x,ax=plt.subplots(1,2,figsize=(10,10))
g = pd.crosstab(df['Month'],df["Day/Nightride"]).plot(kind='bar',ax=ax[0])
plt.axhline(z,color='g',linestyle='--',label='Mean number of travels across months')
sns.countplot(df['Month'],ax=ax[1])
plt.legend()


# ### From the above data we can see the number of rides for each month and number of day and night rides for each month

# In[ ]:


a=df.groupby('Month')['START*'].count()
b=a.index
plt.plot(b,df.groupby('Month')['START*'].count())


# The above plot shows the trend of number of rides as per month
