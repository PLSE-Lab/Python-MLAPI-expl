#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[ ]:


# read the file from the current directory
data=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


#check if the file was successfully downloaded and check for its first 5 elements 
data.head()


# In[ ]:


data.info()


# In[ ]:


data.shape
# Gives us the data entries with element in the tupple() as the number of rows and 2nd element as the number of columns 


# Now since we know we have data types as float64, int64, object, we are on the lookout of those with object dtypes as this will have data with possible Nan values(missing values)

# In[ ]:


# data['name'].str.lower().head()
#lets see how many missing values are there in our data 
data.isna().sum()


# Seems that we have couple of colums with many missing values. Careful and do not be in hurry to remove the missing value, because mostly it is a key indicator in predicting and analysing data. So hold on to it for now!

#  since we have already gathered information about the data types above 
#  we shall skip for explicit data view,howevernever miss to precisely check for datatype in each column
#  data.dtypes

# **** Now lets perform some statistical description on data. Remember doing statistical rehearsal on data always reveals some very important paradigm hidden in our data set****

# In[ ]:


data.describe()


# Key Information from this:
# 1. Since price mean is only $ 152.72 and max price is $10,000, the mean value of the observation lies between the median and the 3rd Quartile.
# 2. The availability ranges from 0 to 365 days with most of the listings available between Q2 and Max of the 365 days.
# 3. There are certain properties which get as many as 629 reviews and as little as 0 so the data is spread .
# 
# Now that we have studied the central tendency of our data through statistical information above, lets check for Outliers in our price column to see the spread
# 

# In[ ]:


Q3=data.price.quantile(0.75)
Q1=data.price.quantile(0.25)
print(Q1)
print(Q3)
IQR=Q3-Q1


# In[ ]:


IQR


# 

# In[ ]:


Outlier_price=(data['price']<(Q1-1.5*IQR))|(data['price']>(Q3+(1.5*IQR)))
print(Outlier_price.value_counts())
data[Outlier_price].price


# Wooh! Thats a lot;these are prices that are affecting our distribtuion of data and can make hard for us to make prediction of our data.Notice that outlier is something that changes the dynamics of research over data.  

# In[ ]:


#Now lets check out the unique categorical values in the column room_type and convert it to numerical.


# In[ ]:


data['room_type'].unique()


# In[ ]:


#Lets drop some columns which as of now are of less use and will be the last thing we want to know in our data analysis 

# data.drop(['host_id','host_name','name'], inplace =True)

data.head()

# Data will now look a bit better.


# In[ ]:


import seaborn as sns 


# In[ ]:


fig=plt.figure(figsize=(20,10))
sns.boxplot(x='neighbourhood_group',y='price',data=data)


# In[ ]:


sns.heatmap(data.corr(),cmap='coolwarm')
plt.title('Correlation')


# Correlation matrix doesn't help in finding the correlation between different columns. Only reviews_per_month is a little correlated to number_of_reviews

# Lets check the neighbourhood_group having highest distribution in data using catplot 

# In[ ]:


sns.catplot(y='neighbourhood_group',
           kind='count',
           height=7,
           aspect=1.5,
           order=data.neighbourhood_group.value_counts().index,
           data=data)


# Clearly, Manhattan has highest listing followed by Brooklyn,Queens,Bronx, & Staten Island

# In[ ]:


sns.catplot(x='room_type',
            kind='count',
           height=7,
           aspect=2,
           data=data,
           color='grey',
           )
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel('room_type',fontsize=40)
plt.ylabel('count',fontsize=40)
plt.title('Distribution as per Room Type',fontsize=40)


# In[ ]:


data['room_type'].unique()
from sklearn  import preprocessing
le=preprocessing.LabelEncoder()
data['room_type']=le.fit_transform(data['room_type'])
data['room_type'].value_counts()


# # This shows to us that the maximum number of properties acquired for renting in Airbnb 
# # is Entire home/apt followed by >> Private room>>and >>Shared room

# In[ ]:


data.head()


# In[ ]:


#We can also see that there are multiple host_name,id, host_id, & name in our data. It will be of less advantage to us in our further knowledge. So we will remove this feature from our 
# data


# In[ ]:


data.drop(['host_name','id','host_id','name'],axis=1,inplace=True)


# In[ ]:


data.groupby('neighbourhood_group').sum()

