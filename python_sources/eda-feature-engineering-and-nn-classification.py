#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis

# ### 1. Exploration
# ### 2. Cleaning
# ### 3. Feature Engineering 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.style.use('fivethirtyeight')


# In[ ]:


data = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


# get overview of numeric columns

data.describe().T


# In[ ]:


data['hotel'].value_counts()/len(data)*100


# #### Checking for class imbalance: 37% - 63%

# In[ ]:


data['is_canceled'].value_counts()/len(data)*100  


# #### Checking for missing values per variable.

# In[ ]:


data.isna().sum()/len(data)*100


# In[ ]:


# drop columns with too many missing values

data.drop(['agent', 'company'], axis=1, inplace=True)


# In[ ]:


# transform target variable into categorical

data['is_canceled'] = pd.Categorical(data['is_canceled'])


# ### Distribution Channel

# In[ ]:


data['distribution_channel'].value_counts()/len(data)


# In[ ]:


print(len(data[(data['distribution_channel'] == 'TA/TO') | (data['distribution_channel'] == 'Direct') | (data['distribution_channel'] == 'Corporate')]))
print(len(data))


# __Only regarding the first three distribution channels only drops about 200 data points. This is equivalent to only 0.1% of the data. Therefore,  we drop the other distribution channels.__

# In[ ]:


data = data[(data['distribution_channel'] == 'TA/TO') |
            (data['distribution_channel'] == 'Direct') |
            (data['distribution_channel'] == 'Corporate')] 


# In[ ]:


data['distribution_channel'].value_counts()


# __Then drop the remaining NAs from the data:__

# In[ ]:


print(len(data))
print(len(data.dropna()))


# In[ ]:


data = data.dropna()


# ### Reserved Room Type

# In[ ]:


# Distribution of variable

data['reserved_room_type'].value_counts()


# __We drop rooms L & P because they barely have data points recorded. We need to drop the room levels from two variables__

# In[ ]:


data = data[(data['reserved_room_type'] != 'P') & (data['reserved_room_type'] != 'L')]
data = data[(data['assigned_room_type'] != 'P') & (data['assigned_room_type'] != 'L')]


# __Check the percentage of cancellations per room type__

# In[ ]:


room_cancel=data.groupby('reserved_room_type')['is_canceled'].value_counts().unstack() 


# In[ ]:


room_cancel=data.groupby('reserved_room_type')['is_canceled'].value_counts().unstack()
room_cancel['total']=room_cancel[0]+room_cancel[1]
room_cancel['percentage_canceled']=round(room_cancel[1]/room_cancel["total"],2)
room_cancel


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='reserved_room_type',hue="is_canceled",data=data,palette='viridis')


# ### Hotel Types

# In[ ]:


plt.figure(figsize=(4,4))
sns.countplot(x='hotel',hue="is_canceled",data=data,palette='viridis')


# __Looks like the City Hotels have more cancellations than Resort Hotels__

# In[ ]:


hotel_cancel=data.groupby('hotel')['is_canceled'].value_counts().unstack()
hotel_cancel['total']=hotel_cancel[0]+hotel_cancel[1]
hotel_cancel['percentage_canceled']=round(hotel_cancel[1]/hotel_cancel["total"],2)
hotel_cancel


# ### Lead Time
# 
# Shows the number of days that elapsed between the entering date of the booking into the PMS and the arrival date.

# In[ ]:


# strong positive skew

data["lead_time"].plot.hist(alpha=0.5,bins=10)
data["lead_time"].describe()


# __Due to this extreme skewed distribution we create 2 binary variables:__
# 
# - Having a big lead time -> booking far in advance -> lead time >= 160 days
# - Having a small lead time -> recent booking -> lead time <= 14 days

# In[ ]:


data['far_in_advance'] = pd.Categorical(np.where(data['lead_time'] >= 160, 1, 0))
data['recent_booking'] = pd.Categorical(np.where(data['lead_time'] <= 14, 1, 0))
data = data.drop('lead_time', axis=1) # drop initial column


# ### Arrival dates

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(x='arrival_date_month',data=data,hue='is_canceled',palette='viridis')


# We see that bookings some months have higher cancellation rates: April, May, June (so just before the peak season). We also see much higher volume of bookings for the summer months compared to winter time.

# ### Length of stay: in week and weekend nights

# In[ ]:


display(data["stays_in_weekend_nights"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(x='stays_in_weekend_nights',data=data,hue='is_canceled',palette='viridis')


# In[ ]:


display(data["stays_in_week_nights"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(x='stays_in_week_nights',data=data,hue='is_canceled',palette='viridis')


# For these variables we clean the data to look at bookings of maximum one full week. The longer bookings could come from the Resort Hotel category, but they represent only 0.027% of the data. 
# 
# Cut outliers :

# In[ ]:


data = data[data.stays_in_week_nights <= 5]
data = data[data.stays_in_weekend_nights <= 2]


# ### Adults

# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x='adults',data=data,hue='is_canceled',palette='viridis')


# __Drop outliers__

# In[ ]:


data = data[data.adults <= 3]


# ### Repeated Guests

# In[ ]:


plt.figure(figsize = (4,4))
sns.countplot(x='is_repeated_guest',data=data,hue='is_canceled',palette='viridis')


# In[ ]:


data['is_repeated_guest'].value_counts()/len(data)*100


# There are barely any repeated guests in the dataset (only 3%). This means, this variable will not be a good predictor to classify canellations and can be dropped.

# In[ ]:


data = data.drop('is_repeated_guest', axis=1)


# ### Booking changes

# How often a guest changed his/her booking options.

# In[ ]:


data["booking_changes"].describe() 


# It can be seen that the majority of guests rarely change booking details.

# In[ ]:


pd.Categorical(data['booking_changes']).value_counts()


# __Creating a binary variable for booking changes__

# In[ ]:


data['changed_booking'] = pd.Series([0 if x == 0 else 1 for x in data['booking_changes']])

# make it categorical
data['changed_booking'] = pd.Categorical(data['changed_booking'])


# ### Children and babies

# In[ ]:


data["children"].value_counts()


# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x='children',data=data,hue='is_canceled',palette='viridis')


# The number of children seem to lead to a slightly different percentage of cancellations. We tried grouping this variable but we obtained a balanced percentage, therefore we just clean the outliers.

# In[ ]:


data = data[data.children <= 2]


# ### Customer Type

# In[ ]:


data["customer_type"].value_counts()


# In[ ]:


plt.figure(figsize = (10,4))
sns.countplot(x='customer_type',data=data,hue='is_canceled',palette='viridis')


# We see that the Transient-Party category has a lower cancellation rate than Transient. 

# ### Deposit Type

# In[ ]:


data["deposit_type"].value_counts()


# In[ ]:


plt.figure(figsize = (6,4))
sns.countplot(x='deposit_type',data=data,hue='is_canceled',palette='viridis')


# In[ ]:


data[data["deposit_type"]=="Non Refund"]["is_canceled"].value_counts()


# This variable shows surprising results: __it seems that for the non-refundable bookings, virtually everyone cancelled.__ We also clean the 'Refundable' category because there is almost no data and this might influence the model just because of the sample distribution.

# In[ ]:


data = data[data.deposit_type != 'Refundable']


# ### Meal

# In[ ]:


data["meal"].value_counts()


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(x='meal',data=data,hue='is_canceled',palette='viridis')


# ### Previous Cancellations

# In[ ]:


data["previous_cancellations"].value_counts()


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(x='previous_cancellations',
              data=data[data["previous_cancellations"]>=1],
              hue='is_canceled',
              palette='viridis')


# Here we see a trend showing that almost everyone who canceled al least once is almost guaranteed to cancel. We will make this variable binary: cancelled before or not

# In[ ]:


data['previous_cancel'] = pd.Series([0 if x == 0 else 1 for x in data['previous_cancellations']]) # binary for previous_cancelations
data['previous_cancel'] = pd.Categorical(data['previous_cancel'])


# ### Room type reserved/assigned

# In[ ]:


sum(data["reserved_room_type"]==data["assigned_room_type"])


# In[ ]:


sum(data["reserved_room_type"]!=data["assigned_room_type"])


# In[ ]:


data[data["reserved_room_type"]!=data["assigned_room_type"]]["is_canceled"].value_counts()


# The guests who are assigned a different room type don't cancel - maybe they got a room upgrade for free. We will make a new binary variable that checks if the guest is assigned a different room type than the one booked.

# In[ ]:


# same room type assigned

bol = data['assigned_room_type'] == data['reserved_room_type']
data['right_room'] = bol.astype(int)
data['right_room'] = pd.Categorical(data['right_room'])


# ### Adr

# In[ ]:


data["adr"].describe()


# In[ ]:


sns.distplot(data["adr"], kde = False)


# We clean the outliers from this column. We see a few points for which the adr (room rate) is 0 - as we don't have extra information about this, we decide to keep them as this could be special promotions or employees who can book for free.

# In[ ]:


data = data[data.adr >=0 ]
data = data[data.adr <= 300]


# In[ ]:


sns.distplot(data["adr"], kde = False)


# ### Special requests

# In[ ]:


data["total_of_special_requests"].value_counts()


# In[ ]:


plt.figure(figsize = (10,4))
sns.countplot(x='total_of_special_requests',
              data=data[data["total_of_special_requests"]>=1],hue='is_canceled',palette='viridis')


# We see that any special requests significantly decreases the cancellation rate - this can show that the guests really made an effort to contact the hotel and their intention to stay. 
# 
# We will make a new binary variable that checks if there are any special requests for a booking:

# In[ ]:


data['special_requests'] = pd.Series([0 if x == 0 else 1 for x in data['total_of_special_requests']]) 
# binary for special requests
data['special_requests'] = pd.Categorical(data['special_requests'])


# ### Reservation status

# In[ ]:


data["reservation_status"].value_counts()


# We will drop this variable as it is the same as the dependent variable 'is_canceled', which groups the 'canceled' and 'no-show' categories.

# In[ ]:


data["is_canceled"].value_counts()


# ### Required car parking

# In[ ]:


data["required_car_parking_spaces"].value_counts()


# In[ ]:


plt.figure(figsize = (10,4))
sns.countplot(x='required_car_parking_spaces',
              data=data,hue='is_canceled',palette='viridis')


# We will also transform this variable to binary: parking space requested or not.

# In[ ]:


data['required_car_parking_spaces'] = pd.Series([0 if x == 0 else 1 for x in data['required_car_parking_spaces']]) 
# binary for parking spots
data['required_car_parking_spaces'] = pd.Categorical(data['required_car_parking_spaces'])


# ## Exploration and Feature Engineering done

# In[ ]:


data = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")


# ## Overview cleaning

# In[ ]:


# encode dependent variable for classification
data['is_canceled'] = pd.Categorical(data['is_canceled']) 

#------------------ Clean the data and transform ---------------------------#

# cut under represented factor levels or split into binary

## Distribution channel

# TA/TO        0.819750
# Direct       0.122665
# Corporate    0.055926
# GDS          0.001617
# Undefined    0.000042

data = data[(data['distribution_channel'] == 'TA/TO') |
            (data['distribution_channel'] == 'Direct') |
            (data['distribution_channel'] == 'Corporate')] 

#---------------------------------#
## Rooms

# A    85446
# D    19161
# E     6470
# F     2890
# G     2083
# B     1114
# C      931
# H      601
# L        6
# P        2

# we clean room types P and L
data = data[(data['reserved_room_type'] != 'P') & (data['reserved_room_type'] != 'L')]
data = data[(data['assigned_room_type'] != 'P') & (data['assigned_room_type'] != 'L')]


#---------------------------------#
## Duration of stay & Kids

# cleaned for now: or create binary with long / short stays
data = data[data.stays_in_weekend_nights <= 2]
data = data[data.stays_in_week_nights <= 5]

# all very unequal distributed
data = data[data.children <= 2]
data = data[data.adults <= 3]

#---------------------------------#
## Deposit

# No Deposit    104641
# Non Refund     14587
# Refundable       162

# also cannot logical combine with other levels
data = data[data.deposit_type != 'Refundable']

#---------------------------------#

# agent                             13.686238 maybe we can keep agent -> only 13 % missing
# company                           94.306893


data = data[data.adr >=0 ]
data = data[data.adr <= 300]

#------------------------Creating new variables-----------------------------------------#


# correct room type assigned
bol = data['assigned_room_type'] == data['reserved_room_type']
data['right_room'] = bol.astype(int)
data['right_room'] = pd.Categorical(data['right_room'])

# lead time
# lead time severly skewed -> binary and we will drop initial variable
data['far_in_advance'] = pd.Categorical(np.where(data['lead_time'] >= 160, 1, 0))
data['recent_booking'] = pd.Categorical(np.where(data['lead_time'] <= 14, 1, 0))


data['changed_booking'] = pd.Series([0 if x == 0 else 1 for x in data['booking_changes']])# binary for changed booking at least once
data['changed_booking'] = pd.Categorical(data['changed_booking'])

data['previous_cancel'] = pd.Series([0 if x == 0 else 1 for x in data['previous_cancellations']]) # binary for previous_cancelations
data['previous_cancel'] = pd.Categorical(data['previous_cancel'])

data['special_requests'] = pd.Series([0 if x == 0 else 1 for x in data['total_of_special_requests']]) # binary for special requests
data['special_requests'] = pd.Categorical(data['special_requests'])

data['required_car_parking_spaces'] = pd.Series([0 if x == 0 else 1 for x in data['required_car_parking_spaces']]) # binary for parking spots
data['required_car_parking_spaces'] = pd.Categorical(data['required_car_parking_spaces'])



# In[ ]:


#--------- drop initial variables for those where we create binaries------------

data = data.drop(['total_of_special_requests',
                   'reservation_status',
                   'previous_cancellations',
                   'booking_changes',
                   'lead_time',
                  'babies'
                    ], axis=1)

# --------- drop columns which we don't want to use --------------

data = data.drop(['company','agent','is_repeated_guest',
                  'reservation_status_date', 'previous_bookings_not_canceled'
                 ],axis=1)

# ---------- drop remaining NAs from the data---------
data=data.dropna()


# In[ ]:


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Bidirectional
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Layer

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from scipy.stats import zscore


# In[ ]:


# drop unused columns
data = data.drop(['arrival_date_week_number', 'arrival_date_day_of_month'], axis=1 )


# In[ ]:


def encode_columns(column, data):
    
    data = pd.concat([data,pd.get_dummies(data[column],prefix=column)],axis=1)
    data.drop(column, axis=1, inplace=True)
    
    return data


# In[ ]:


### ------------- encode categorical columns ----------------

categorical_columns = ["required_car_parking_spaces",
                       "right_room",
                       "far_in_advance",
                       "recent_booking",
                       "changed_booking",
                       "previous_cancel",
                       "special_requests",
    
    
                       "hotel", 
                       "arrival_date_year",
                       "arrival_date_month",
                       "meal",
                       "country",
                       "market_segment",
                       "distribution_channel",
                       "deposit_type",
                       "customer_type",
                       "reserved_room_type",
                       "assigned_room_type"
                      ]
    
for col in categorical_columns:
    data=encode_columns(col,data)


# In[ ]:


data['adr'] = zscore(data['adr'])


# In[ ]:


data = data.dropna()


# In[ ]:


x = data.drop('is_canceled', axis=1)
y = data['is_canceled']


# In[ ]:


x = np.asarray(x)
y = np.asarray(y)


# In[ ]:


# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(100, input_dim=x.shape[1], activation='relu', kernel_initializer='random_normal'))
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu',kernel_initializer='random_normal'))
model.add(Dropout(0.2))
model.add(Dense(25,activation='relu',kernel_initializer='random_normal'))
model.add(Dense(1,activation='sigmoid', kernel_initializer='random_normal'))


# compile the model
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=12, 
                        verbose=1, mode='auto', restore_best_weights=True)


history = model.fit(x_train, y_train, validation_split=0.2, callbacks=[monitor], verbose=1, epochs=100)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Accuracy: %f' % (accuracy*100))
print('\n')


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve, auc


# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,7))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


prediction_proba = model.predict(x_test)


# In[ ]:


plot_roc(prediction_proba,y_test)

