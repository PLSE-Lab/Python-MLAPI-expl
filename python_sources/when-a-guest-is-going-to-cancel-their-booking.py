#!/usr/bin/env python
# coding: utf-8

# In this kernel I am trying to predict whether or not a guest is going to cancel their booking. I did some data exploration and remove some variables from the analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataframe = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')


# In[ ]:


use_columns = list(dataframe.columns)


# In[ ]:


use_columns.remove('previous_cancellations')


# In[ ]:


pd.set_option('display.max_columns', None)
display(dataframe.head())
pd.set_option('display.max_columns', 10)


# In[ ]:


dataframe.info()


# Company is the one with the biggest number of null values, but as we can see this means that the guests are not doing the reservation thought a company. Let fill the company attribute with 0. The same happen to 'agent'.

# # company and agent

# In[ ]:


dataframe['company'] = dataframe['company'].fillna('no company')
dataframe['agent'] = dataframe['agent'].fillna('no agent')


# In[ ]:


display(f"Number of unique companies '{dataframe.company.unique().shape[0]}' ")
display(f"Number of unique agent '{dataframe.agent.unique().shape[0]}' ")


# Since the number of unique values of agent and companies are extremily big, let's replace the values for 0 and 1. This means if the guest uses company, it will be 1 and in case it does not use it, it will be 0. The same procesedure to agent

# In[ ]:


dataframe.loc[dataframe['company'] != 'no company', 'company'] = 1
dataframe.loc[dataframe['company'] == 'no company', 'company'] = 0


# In[ ]:


dataframe.loc[dataframe['agent'] != 'no agent', 'agent'] = 1
dataframe.loc[dataframe['agent'] == 'no agent', 'agent'] = 0


# # Let's see if the data is well balance

# In[ ]:


sns.countplot(dataframe['is_canceled'])
plt.title("Number of cancel booking and not cancel booking")
plt.show()


# The data seems to be balance. Let's move on

# # arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month

# Since I am not doing any temporal analysis, I am checking all the other features.

# In[ ]:


use_columns.remove('arrival_date_year')
use_columns.remove('arrival_date_month')
use_columns.remove('arrival_date_week_number')
use_columns.remove('arrival_date_day_of_month')


# # hotel

# hotel influence on the cancelation?

# In[ ]:


pd.DataFrame(dataframe.groupby(['is_canceled', 'hotel']).size())


# Dont think so

# In[ ]:


use_columns.remove('hotel')


# # reserved_room_type

# In[ ]:


dataframe.groupby(['assigned_room_type','is_canceled']).size()


# In[ ]:


dataframe.groupby(['reserved_room_type','is_canceled']).size()


# In[ ]:


use_columns.remove('reserved_room_type')
use_columns.remove('assigned_room_type')


# # deposit_type

# In[ ]:


dataframe.groupby(['deposit_type','is_canceled']).size()


# As we can see above, this variable definitely helps to predict the model. So, the chance of someone who did a deposit like Non Refund cancel a booking is huge 

# # days_in_waiting_list

# The numbers of days in the waiting list have any influence on the cancelation?

# In[ ]:


pd.DataFrame(dataframe.groupby(['is_canceled'])['days_in_waiting_list'].mean())


# In[ ]:


pd.DataFrame(dataframe.groupby(['is_canceled'])['days_in_waiting_list'].std())


# In[ ]:


pd.DataFrame(dataframe.groupby('is_canceled')['days_in_waiting_list'].median())


# Looking only for the mean, it seems that the number of days have some influence; however, looking for the std, we can see that the values vary a lot. When we analyse the median, we can see that most part of the people cancel their book even when there are not in the waiting list.

# If we analyse the values below, we indeed can see, that the days_in_waiting_list does not influence the is_canceled

# In[ ]:


use_columns.remove('days_in_waiting_list')


# # total_of_special_requests

# Does the number of total_of_special_requests in a reservation influence the cancelation rate?

# In[ ]:


dataframe.groupby(['total_of_special_requests', 'is_canceled']).size()


# Looking for the above data, we can see that the number of requests does have influence on the cancelation rate.

# # required_car_parking_spaces

# Does the number of required_car_parking_spaces in a reservation influence the cancelation rate?

# In[ ]:


dataframe.groupby(['is_canceled', 'required_car_parking_spaces']).size()


# This variables helps a lot. When a booking is created and the guest request at least one space, he is definily going to hotel

# # reservation_status e reservation_status_date

# > This variables are used to show whether or not a reservation was made or not. This can be used to futher analysis, but not for predict the is_canceled

# In[ ]:


use_columns.remove('reservation_status_date')
use_columns.remove('reservation_status')


# # Finally, let's do some prediction

# In[ ]:


labelEncoder = LabelEncoder()
encoded_dataframe = dataframe[use_columns].apply(lambda x: labelEncoder.fit_transform(x.astype(str)))


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(encoded_dataframe.drop('is_canceled', axis=1), encoded_dataframe['is_canceled'])


# In[ ]:


model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


confusion_matrix(y_pred, y_test)


# In[ ]:


pd.DataFrame.from_dict([dict(zip(encoded_dataframe.drop('is_canceled', axis=1), model.feature_importances_))]).T.sort_values(0)


# As we can see, 6 features does not aggregate too much to the model, so lets remove than and retrain

# In[ ]:


pd.DataFrame.from_dict([dict(zip(encoded_dataframe.drop('is_canceled', axis=1), model.feature_importances_))]).T.sort_values(0)[:6].index


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(encoded_dataframe.drop(['babies', 'is_repeated_guest', 'company',
       'previous_bookings_not_canceled', 'agent', 'children', 'is_canceled'], axis=1), encoded_dataframe['is_canceled'])


# In[ ]:


model.fit(x_train, y_train)
y_pred = model.predict(x_test)
confusion_matrix(y_pred, y_test)


# In[ ]:


pd.DataFrame.from_dict([dict(zip(x_train, model.feature_importances_))]).T.sort_values(0)

