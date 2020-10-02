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


# ## Importing Libraries & Reading Dataset

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes

#Reading all data files....
rates = pd.read_csv('/kaggle/input/ratedetails/rates.csv')
reservations = pd.read_csv('/kaggle/input/reservations/reservations.csv')


# ### 1.a. What are the popular choices of booking rates ?

# In[ ]:


# Join rates and reservation databases on basis of RateId
reservation_with_rates= pd.merge(left=reservations, right=rates, left_on='RateId', right_on='RateId')
# To find Day of Week using given StartUTC.
reservation_with_rates['StartUtc'] = pd.to_datetime(reservation_with_rates['StartUtc'])
reservation_with_rates['Day of Week'] = reservation_with_rates['StartUtc'].dt.weekday_name
# Plot popular choices of booking Rates Names
fig = px.histogram(reservation_with_rates, x="RateName")
fig.show()


# ### Explanation :
# We can observe that, Fully Flexible is the most popular rate choice among customer, then followed by Early - 60 days rooms.

# ### 1.b. What are the popular choices of booking rates for different segments of customers ?

# In[ ]:


#  Plot booking rate choice with different Customer Segment (eg. AgeGroup, Gender, NationalityCode)
x=["AgeGroup","Gender","NationalityCode"]
for i in x:
    print(i,"v/s RateName")
    test5 = reservation_with_rates.groupby(['RateName',i])[i].count().unstack('RateName').plot(kind='bar', stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gcf().set_size_inches(15, 10)
    plt.show()


# 
# ### Explanation
# ### For Age Group v/s Rate Name
# #### In all the age groups we can observe that, customers are more likely to book rooms with Fully Flexible rate than any other rates.
# #### Moreover, age group 0 contribute to more than 50% of customer bookings for given hotel property.
# ### For Gender v/s Rate Name
# #### In all the gender groups we can observe that, customers are more likely to book rooms with Fully Flexible rate followed by Early - 60 days rate and then by Early - 21 days rate compared to other rates.
# #### Also, gender group 1 contribute to about 50% of customer bookings for given hotel property whereas gender group 2 contributes to least number of customers.
# ### For Nationality v/s Rate Name
# #### Here we can conclude that customers from most of the nationalities are tend to book room with Fully Flexible rate than rooms with any other rate. 
# #### In Addition, people with nationalities of United States of America(USA), Great Britain (GB) and Denmark (DE) are more likely to book rooms with the given hotel property.

# ### 2.a. What are the typical guests who pursue online check-in?

# In[ ]:


# selecting guest data who pursue online check-in using column IsOnlineCheckin
online_reservation_with_rates = reservation_with_rates[reservation_with_rates.IsOnlineCheckin == 1]
# Plot typical guests who pursue online check-in using BusinessSegment column.
fig = px.histogram(online_reservation_with_rates, x="BusinessSegment")
fig.show()


# ### Explanation
# Customers having business segments OTAs followed by OTA Netto and Leisure are typical guests who did online check-in.

# ## 2.b. Does it vary across week days?
# 

# In[ ]:


# Plot variation in typical guests who pursue online check-in using BusinessSegment and Day of Week columns.
print("Guest with Online Check-in v/s Business Segment v/s Weekday")
test = online_reservation_with_rates.groupby(['BusinessSegment','Day of Week'])['Day of Week'].count().unstack('BusinessSegment').plot(kind='bar', stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(15, 10)
plt.show()


# ### Explanation
# ### The bookings varies across the weekdays, following are the observations:
# - The bookings completed through **_OTAs_** are highest throughout the week except on **_Tuesdays and Wednesdays_** where bookings through **_OTA Netto_** is the highest.
# - Also, the rooms bookings are highest on **_Fridays_**. Whereas there are no room bookings through **_Direct Business_** and **_FIT_** on Weekends(i.e. Saturdays and Sundays).
# - Customer with **_FIT_** room bookings tend to visit hotels only on **_Tuesdays, Wednesdays and Fridays_**.

# ###  Alternate Solution
# #### If we consider that typical guest is the one who does not cancel the booking, then the below mentioned dataframe contains typical guests.

# In[ ]:


# selecting guest data having null value in CancellationReason and 1 in IsOnlineCheckin column
non_Cancelled_reservation_with_rates = online_reservation_with_rates[online_reservation_with_rates.CancellationReason.notnull()]
non_Cancelled_reservation_with_rates.head()


# 
# 
# ## 3.  Look at the night cost per occupied space. What guest segment is  the most profitable per occupied space unit? And what guest  segment is the least profitable? 

# In[ ]:


# Select dataframe which best describes guest segment from given data.
guest_Segment_df=reservation_with_rates[["AgeGroup","Gender","NationalityCode","BusinessSegment","CancellationReason"]]

# Column NationalityCode has 1096 rows with Null values. Replacing these null values with value 'Other'.
guest_Segment_df['NationalityCode'].fillna("Other",inplace=True)
# Column CancellationReason code has 1806 rows with Null values. Replacing these null values with value '100'.
guest_Segment_df['CancellationReason'].fillna(100,inplace=True)

# Categorical encoding of NationalityCode column. Converting categorical values to binary values using one hot encoding method
guest_Segment_df['NationalityCode'] = pd.Categorical(guest_Segment_df['NationalityCode'])
dfDummies_Nationality = pd.get_dummies(guest_Segment_df['NationalityCode'], prefix = 'nationality')
Nationality_guest_Segment_df = pd.concat([guest_Segment_df, dfDummies_Nationality], axis=1)

# Categorical encoding of BusinessSegment column. Converting categorical values to binary values using one hot encoding method
Nationality_guest_Segment_df['BusinessSegment'] = pd.Categorical(Nationality_guest_Segment_df['BusinessSegment'])
dfDummies_Business = pd.get_dummies(Nationality_guest_Segment_df['BusinessSegment'], prefix = 'business')
business_Segment_guest_Segment_df = pd.concat([Nationality_guest_Segment_df, dfDummies_Business], axis=1)

# Encoding of CancellationReason column. Converting ordinal values (0<1<2<3..) values to binary values (1s and 0s) using one hot encoding method
business_Segment_guest_Segment_df['CancellationReason'] = pd.Categorical(business_Segment_guest_Segment_df['CancellationReason'])
dfDummies_cancellation = pd.get_dummies(business_Segment_guest_Segment_df['CancellationReason'], prefix = 'cancellation')
cancellation_guest_Segment_df = pd.concat([business_Segment_guest_Segment_df, dfDummies_cancellation], axis=1)

# Encoding of Gender column. Converting ordinal values (0<1<2) values to binary values (1s and 0s) using one hot encoding method
cancellation_guest_Segment_df['Gender'] = pd.Categorical(cancellation_guest_Segment_df['Gender'])
dfDummies_gender = pd.get_dummies(cancellation_guest_Segment_df['Gender'], prefix = 'gender')
encoded_guest_Segment_df = pd.concat([cancellation_guest_Segment_df, dfDummies_gender], axis=1)

labelencoder = LabelEncoder()
encoded_guest_Segment_df['AgeGroup'] = labelencoder.fit_transform(encoded_guest_Segment_df['AgeGroup'])
encoded_guest_Segment_df = encoded_guest_Segment_df.drop(["BusinessSegment", "NationalityCode", "CancellationReason","Gender"], axis=1)


# ### Implementing K-mode clustering method to determine customer segments and using elbow method to determine number of clusters 
# - Reason for using K-mode: As the given data is categorical, hence K-mode clustering method is better than K-means clustering method.

# In[ ]:


distortions = []
K = range(1,10)
for k in K:
    kmodeModel = KModes(n_clusters=k).fit(encoded_guest_Segment_df)
    kmodeModel.fit(encoded_guest_Segment_df)
    distortions.append(sum(np.min(cdist(encoded_guest_Segment_df, kmodeModel.cluster_centroids_, 'hamming'), axis=1)) / encoded_guest_Segment_df.shape[0])

# Plot the elbow graph
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[ ]:


# Scaling all the features in the dataframe to normalize the data in a particular range. 
# Also number of clusters are 4 (derived from elbow method)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(encoded_guest_Segment_df)
kmodes = KModes(n_clusters=4, random_state=0) 
y = kmodes.fit_predict(X_scaled)
reservation_with_rates['Cluster_Segment'] = y

# Grouping NightCost_Sum and OccupiedSpace_Sum on the basis of Cluster_Segment
nightcost_Sum_reservation_with_rates = reservation_with_rates.groupby("Cluster_Segment",as_index=False)["NightCost_Sum","OccupiedSpace_Sum"].sum()
# Calculating Night Cost per occupied space from above dataframe
nightcost_Sum_reservation_with_rates['Night Cost per occupied space'] = nightcost_Sum_reservation_with_rates['NightCost_Sum']/nightcost_Sum_reservation_with_rates['OccupiedSpace_Sum']
profit_reservation_with_rates = nightcost_Sum_reservation_with_rates.sort_values('Night Cost per occupied space')

# Plotting Night Cost per occupied space v/s Cluster_Segment (Most and Least profitable Guest Segment)
fig = px.bar(profit_reservation_with_rates, x="Cluster_Segment", y="Night Cost per occupied space", orientation='v')
fig.show()


# - Customers of segment 2 have highest night cost per occupied space. 
# - Customers of segment 3 have least night cost per occupied space. 
# 
# ### Moreover, we can use this analysis for further prediction like by using this data, we can classify our future customer into these segments and we can act accordingly for taking business decision.
