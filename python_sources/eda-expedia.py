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
train = pd.read_csv('/kaggle/input/expedia-hotel-recommendations/train.csv', usecols = ['user_id'])


# In[ ]:


users = train['user_id'].unique()


# In[ ]:


users.shape


# In[ ]:


import numpy as np
select_users = np.random.choice(users, 100000, replace=False)
select_users.shape


# In[ ]:


select_users


# In[ ]:


y = pd.DataFrame(select_users)
y.columns = ['id']
y


# In[ ]:


import pandas as pd
path = '/kaggle/input/expedia-hotel-recommendations/train.csv'
iter_csv = pd.read_csv(path, iterator=True, chunksize=1000)
select = pd.concat([chunk.loc[chunk.user_id.isin(y['id'])] for chunk in iter_csv])
select.head()


# In[ ]:


df = select


# In[ ]:


df.shape


# **EDA**
# Start visualization
# 
# 1. timeline
# 1. click vs booking
# 1. Destination country overlay Destimation booking

# In[ ]:


df.columns


# In[ ]:


df['date'] = pd.to_datetime(df['date_time'])
df['date'] = df['date'].dt.date
df['month-year'] = pd.to_datetime(df['date']).dt.to_period('M')
df['hours'] = pd.to_datetime(df['date_time'])
df['hours'] = df['hours'].dt.hour
df = df.sort_values(by = 'month-year')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(30,20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(2, 2, 1)
sns.countplot(df['posa_continent'], ax=ax)

ax = fig.add_subplot(2, 2, 2)
sns.countplot(df['month-year'], ax=ax)
plt.xticks(rotation=90)

ax = fig.add_subplot(2, 2, 3)
sns.countplot(df['hotel_cluster'], ax=ax)
plt.xticks(rotation=90)

ax = fig.add_subplot(2, 2, 4)
sns.countplot(df['is_booking'], ax=ax)
plt.show()


# In[ ]:


sns.barplot(df['hotel_cluster'], hue =df['is_booking'])
plt.xticks(rotation=90)


# In[ ]:


sns.countplot(x=df['hotel_cluster'], hue=df['is_booking'])
plt.xticks(rotation=90)


# In[ ]:


sns.countplot(df['is_booking'])
plt.show()


# In[ ]:


sns.countplot(df['month-year'],hue=df['is_booking'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(30,20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(2, 2, 1)
sns.countplot(df['hours'], ax=ax)
ax = fig.add_subplot(2, 2, 2)
sns.countplot(df['is_package'], ax=ax)
plt.xticks(rotation=90)
ax = fig.add_subplot(2, 2, 3)
sns.countplot(df['channel'], ax=ax)
plt.xticks(rotation=90)
ax = fig.add_subplot(2, 2, 4)
sns.distplot(df['srch_adults_cnt'], ax=ax)
plt.show()


# In[ ]:


sns.FacetGrid(df, hue="is_booking", size=6)    .map(plt.hist, "hotel_cluster")    .add_legend()
plt.title('book or click')
plt.show()


# In[ ]:


df['day'] = pd.to_datetime(df['date']).dt.day
df['month'] = pd.to_datetime(df['date']).dt.month


# In[ ]:


df.head().transpose()


# In[ ]:


df.columns


# In[ ]:


df_1 = df[['site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'user_id', 'is_mobile', 'is_package',
       'channel', 'srch_destination_id', 'srch_destination_type_id',
       'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
       'hotel_cluster', 'day', 'month', 'hours']]


# In[ ]:


x = pd.DataFrame(df_1.groupby(['user_location_country']).size())
x.transpose()


# In[ ]:


df_1['is_clicking'] = df_1['is_booking']
df_1.is_clicking[df_1.is_clicking == 0] = 2
df_1.is_clicking[df_1.is_clicking == 1] = 0
df_1.is_clicking[df_1.is_clicking == 2] = 1


# In[ ]:


df_1[['is_booking', 'is_clicking']].head()


# In[ ]:


df_2 = df_1.groupby(['user_id', 'hotel_cluster', 'site_name', 'posa_continent', 'user_location_country','channel', 'hotel_continent', 'hotel_country', 'hotel_market']).sum()[['is_booking', 'is_clicking', 'is_mobile', 'is_package', 'cnt']].reset_index()
df_2.head


# In[ ]:


df_2.tail(20).transpose()


# In[ ]:


df_2.head(20).transpose()


# In[ ]:





# In[ ]:





# In[ ]:


df_1.user_location_country[df_1.user_location_country == "NaN"] = 1000001
df_1.user_location_region[df_1.user_location_region == "NaN"] = 1000001
df_1.user_location_city[df_1.user_location_city == "NaN"] = 1000001
df_1.srch_destination_id[df_1.srch_destination_id == "NaN"] = 1000001
df_1.srch_destination_type_id[df_1.srch_destination_type_id == "NaN"] = 1000001
df_1.user_location_country[df_1.user_location_country == "NaN"] = 1000001
df_1.hotel_continent[df_1.hotel_continent == "NaN"] = 1000001
df_1.hotel_country[df_1.hotel_country == "NaN"] = 1000001
df_1.hotel_market[df_1.hotel_market == "NaN"] = 1000001


# In[ ]:


site_name = pd.get_dummies(df_1["site_name"], prefix = 'site_name: ')
posa_continent = pd.get_dummies(df_1["posa_continent"], prefix = 'posa_continent: ')
user_location_country = pd.get_dummies(df_1["user_location_country"], prefix = 'user_location_country: ')
user_location_region = pd.get_dummies(df_1["user_location_region"], prefix = 'user_location_region: ')
#user_location_city = pd.get_dummies(df_1["user_location_city"], prefix = 'user_location_city: ')
#srch_destination_id = pd.get_dummies(df_1["srch_destination_id"], prefix = 'srch_destination_id: ')
srch_destination_type_id = pd.get_dummies(df_1["srch_destination_type_id"], prefix = 'srch_destination_type_id: ')
hotel_continent = pd.get_dummies(df_1["hotel_continent"], prefix = 'hotel_continent: ')
hotel_country = pd.get_dummies(df_1["hotel_country"], prefix = 'hotel_country: ')
#hotel_market = pd.get_dummies(df_1["hotel_market"], prefix = 'hotel_market: ')


# In[ ]:


df_2 = pd.concat([df_1.drop(['site_name', 'posa_continent',"user_location_country","user_location_region","srch_destination_type_id", "hotel_continent", "hotel_country"], axis = 1), 
                  site_name, posa_continent, user_location_country, user_location_region, srch_destination_type_id,
                 hotel_continent, hotel_country], axis = 1)


# In[ ]:


df_2.head()


# In[ ]:


from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

X = df_2.drop(['hotel_cluster'] , axis=1) 
y = df_2['hotel_cluster'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8) 
  
# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
print (accuracy) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, gnb_predictions) 


# In[ ]:





# In[ ]:





# In[ ]:




