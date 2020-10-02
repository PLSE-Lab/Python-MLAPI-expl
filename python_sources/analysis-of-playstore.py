#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data Cleaning

# In[ ]:


df=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")


# In[ ]:


df.head()


# In[ ]:


df.dropna(subset=['Rating'],inplace=True) #remove all record in which rating is null


# In[ ]:


df[df['Android Ver'].isnull()] #last record has all wrong data in their column


# In[ ]:


df.drop(index=df[df['Size']=='1,000+'].index,inplace=True)


# In[ ]:


def price(x):
    if x[-1]=='M':
        return float(x[:-1])*100000
    elif x[-1]=='k':
        return float(x[:-1])*1000
df['Size']=df['Size'].apply(lambda x: price(x)) #this will remove dollor sign


# In[ ]:


df.dtypes


# In[ ]:


#df.drop(index=10472,inplace=True) #so i removed the last record


# In[ ]:


from statistics import mode
df['Android Ver'].fillna(value=mode(df['Android Ver']),inplace=True)


# In[ ]:


df.loc[[4453,4490]] #after using mode


# In[ ]:


df[df['Current Ver'].isnull()] #same for current version 
#we will not drop any as all record looks fine


# In[ ]:


df['Current Ver'].fillna(value=mode(df['Current Ver']),inplace=True) #fill null with most common value


# In[ ]:


df.loc[[15,1553,6322,7333]]


# In[ ]:


df.tail(5)


# In[ ]:


df.describe()


# In[ ]:


df.dtypes #data types of all column here we can see that some contain numeric data but has object data type
#but first lets clean the price which has dollor sign and object format


# In[ ]:


def price(x):
    if x=='0':
        return x
    else:
        return x[1:]
df['Price']=df['Price'].apply(lambda x: price(x)) #this will remove dollor sign


# In[ ]:


df['Price']=df['Price'].astype('float')


# In[ ]:


#df[df['Installs'].isnull()]
def install(x):
    x=x.split('+')
    y=x[0].split(',')
    return "".join(y)

df['Installs']=df['Installs'].apply(lambda x: install(x)) #this will remove comma and + sign


# In[ ]:


df.head()


# In[ ]:


df['Installs']=df['Installs'].astype('int') #converted to integer


# In[ ]:


df['Reviews']=df['Reviews'].astype('int') #convert reviews to integer


# In[ ]:


df.dtypes #all are converted to right format


# In[ ]:


df[df['Rating']>5] #no rating is greater than 5


# In[ ]:


df[df['Reviews']>df['Installs']] #Reviews should not be more than installs as only those who installed can review the
#app


# In[ ]:


df.drop(index=df[df['Reviews']>df['Installs']].index,inplace=True)


# In[ ]:


df[df['Reviews']>df['Installs']]


# In[ ]:


sns.scatterplot(x='Price',y='Price',data=df) #most of the data are between 0-50. We can't consider the data with price 
#more than 50 as they are our outliers that can affect our analysis


# In[ ]:


#sns.distplot(df['Price'],bins=5)


# In[ ]:


df.drop(index=df[df['Price']>=30].index,inplace=True)


# In[ ]:


sns.scatterplot(x='Price',y='Price',data=df) #without outliers


# In[ ]:


sns.distplot(df['Reviews'],bins=5) #so most of the data are less than 1 million reviews so we remove all above it


# In[ ]:


df.drop(index=df[df['Reviews']>=1000000].index,inplace=True)


# In[ ]:


sns.scatterplot(x='Reviews',y='Reviews',data=df) #without outliers


# In[ ]:


df['Installs'].quantile(q=0.95) #checking the 95th percentile of the data to remove the outliers i.e. last 5s


# In[ ]:


df.drop(index=df[df['Installs']>10000000].index,inplace=True)  #removing last 5% which are outliers


# In[ ]:


df.to_csv('play_store_cleaned.csv')


# ### Data Analysis Starts From Here

# In[ ]:


sns.distplot(df['Rating'])


# Distribution of Ratings is a Left skewed distribution (Mean is smaller than median)
# as Most of the ratings are between 4-4.5 so most of the app perform well
# 
# We should use some other variable with rating to get clear idea of the performance of the applications 
# 

# In[ ]:


df['Content Rating'].value_counts() #as content rating is very less for adults and unrated as they wont needed for
#analysis


# In[ ]:


df.drop(index=df[df['Content Rating']=='Adults only 18+'].index,inplace=True)


# In[ ]:


df.drop(index=df[df['Content Rating']=='Unrated'].index,inplace=True)


# In[ ]:


df['Content Rating'].value_counts()


# In[ ]:


#sns.jointplot(x='Rating',y='Size',data=df)


# we can see from the graph that Most of the app are from 15mb to 40mb and as we know all the rating of apps are quiet well of 4-4.5

# In[ ]:


sns.jointplot(x='Price',y='Rating',data=df)


# Here we can say that paid apps are mostly highly rated
# Exceptional is there of 0 which are free apps which has mixed ratings
# 
# Let us remove the free apps and observe only the paid apps

# In[ ]:


df1=df[df['Price']>0.0]


# In[ ]:


sns.jointplot(x='Price',y='Rating',data=df1)


# As the free apps removed we get a clear idea and here we can conclude that paid apps are highly rated

# In[ ]:


df2=df[['Reviews', 'Size', 'Rating', 'Price']]


# In[ ]:


sns.pairplot(df2)


# Tere is some pattern in size vs price let us explore that

# In[ ]:


sns.jointplot(x='Price',y='Size',data=df)


# Most of the priced items are of lesser size so we can say that People prefer to pay for app of lesser size or app are designed in such a way that they are not bulky
# 
# We can conclude from this is that bulky app may not perform well in the market

# In[ ]:


from statistics import mode
plt.figure(figsize=(5,5))
sns.barplot(x='Content Rating',y='Rating',data=df)


# We can see that at an average all the ratings are nearly the same (close to 4) irrespective of the content rating
# 
# But in mature 17+ it looks like they are lower than others

# In[ ]:


sns.jointplot(x='Rating',y='Reviews',data=df)


# Seems like well known apps(more reviews) gets higher ratings 

# In[ ]:


# Insights


# 1. Distribution of Ratings is a Left skewed distribution (Mean is smaller than median) as Most of the ratings are between 4-4.5 so most of the app perform well
# 2. Most of the app are from 15mb to 40mb and as we know all the rating of apps are quiet well of 4-4.5
# 3. As the free apps removed we get a clear idea and here we can conclude that paid apps are highly rated
# 4. Most of the priced items are of lesser size so we can say that People prefer to pay for app of lesser size or app are designed in such a way that they are not bulky
# 5. We can conclude from this is that bulky app may not perform well in the market
# 6. We can see that at an average all the ratings are nearly the same (close to 4) irrespective of the content rating
# 7. Mature 17+ it looks like they are lower than others
# 8. Well known apps(more reviews) gets higher ratings 

# 
