#!/usr/bin/env python
# coding: utf-8

# # **Classification of Hotel Classes (Resort Hotel, City Hotel)**

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix , classification_report, roc_curve, auc


# In[ ]:


ds = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
ds


# In[ ]:


ds.isnull().sum()


# In[ ]:


ds.dropna(axis = 0, subset=['country', 'children'], inplace=True)
moa = ds['agent'].mean()
ds['agent'].fillna(value=moa, axis=0, inplace=True)
ds.fillna(method='pad', inplace=True)
ds.dropna(inplace=True, subset=['company'])
ds.isnull().sum()


# In[ ]:


ds.info()


# # EDA

# In[ ]:


plt.hist(ds['hotel'], bins = 3, edgecolor = 'k')
plt.title('Hotel Name', fontsize = 20)


# In[ ]:


plt.bar(['0','1'], ds['is_canceled'].value_counts(), edgecolor = 'k', color = 'orange')
plt.title('Is_Canceled', fontsize = 20)


# In[ ]:


plt.bar(['BB', 'HB', 'FB', 'SC', 'Undefined'], ds['meal'].value_counts(), edgecolor = 'k', color = 'green')
plt.title('Meals', fontsize = 20)


# In[ ]:


a = ds['country'].value_counts()
a = a[: 6]

#plt.hist(a , bins = 20, edgecolor = 'k', color = 'gray')
#plt.title('Country', fontsize = 10)
plt.bar(['PRT','GBR','FRA','ESP','DEU','ITA'], a, color = 'red', edgecolor = 'k')
plt.title('Top 6 Countries', fontsize = 20)


# In[ ]:


plt.barh(['Corporate', 'Direct', 'Online TA', 'Offline TA/TO','Complementary', 'Groups', 'Aviation'], ds['market_segment'].value_counts(), color = 'red', edgecolor = 'k')
plt.title('Market Segment', fontsize = 20)


# In[ ]:


plt.bar(['Check-Out','Canceled','No-Show'], ds['reservation_status'].value_counts(), color = 'pink', edgecolor = 'k')
plt.title('Reservation Status', fontsize = 20)


# # Data Preprocessing

# In[ ]:


from sklearn.preprocessing import LabelEncoder

hotel = LabelEncoder()
arrival_date_month = ()
meal = LabelEncoder()
country = LabelEncoder()
market_segment = LabelEncoder()
distribution_channel = LabelEncoder()
reserved_room_type = LabelEncoder()
assigned_room_type = LabelEncoder()
deposit_type = LabelEncoder()
customer_type = LabelEncoder()
reservation_status = LabelEncoder()
reservation_status_date = LabelEncoder()


# In[ ]:


ds['hotel_n'] = hotel.fit_transform(ds['hotel'])

ds['arrival_date_month_n'] = hotel.fit_transform(ds['arrival_date_month'])

ds['meal_n'] = hotel.fit_transform(ds['meal'])

ds['country_n'] = hotel.fit_transform(ds['country'])

ds['market_segment_n'] = hotel.fit_transform(ds['market_segment'])

ds['distribution_channel_n'] = hotel.fit_transform(ds['distribution_channel'])

ds['reserved_room_type_n'] = hotel.fit_transform(ds['reserved_room_type'])

ds['assigned_room_type_n'] = hotel.fit_transform(ds['assigned_room_type'])

ds['deposit_type_n'] = hotel.fit_transform(ds['deposit_type'])

ds['customer_type_n'] = hotel.fit_transform(ds['customer_type'])

ds['reservation_status_n'] = hotel.fit_transform(ds['reservation_status'])

ds['reservation_status_date_n'] = hotel.fit_transform(ds['reservation_status_date'])


# In[ ]:


ds.drop(['hotel','arrival_date_month','meal','country','market_segment','distribution_channel','reserved_room_type','assigned_room_type','deposit_type','customer_type','reservation_status','reservation_status_date'], axis = 1, inplace=True)
ds.info()


# # Data Balancing

# In[ ]:


ds.shape


# In[ ]:


ds['hotel_n'].value_counts()


# In[ ]:


ds_0 = ds[ds['hotel_n'] == 0]
ds_1 = ds[ds['hotel_n'] == 1]

ds_0.shape , ds_1.shape


# In[ ]:


ds_0 = ds_0.sample(ds_1.shape[0])
ds = ds_0.append(ds_1, ignore_index=True)
ds.shape


# In[ ]:


ds['hotel_n'].value_counts()


# # Removing Constant

# In[ ]:


x = ds.drop('hotel_n', axis = 1)
y = ds['hotel_n']


# In[ ]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


filter = VarianceThreshold()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[ ]:


x_train = filter.fit_transform(x_train)
x_test = filter.transform(x_test)

x_train.shape , x_test.shape


# # Standardizing and converting into Numpy array

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# # ML Models

# In[ ]:


xg = XGBClassifier()
xg.fit(x_train,y_train)
xg.score(x_test, y_test)


# In[ ]:


rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf.score(x_test, y_test)


# # Confusion Matrix
# 

# In[ ]:


y_pred = xg.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')


# In[ ]:


print(classification_report(y_test,y_pred))

