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


# ## Hotel Bookings - Feature engineering and Classification

# ### Importing the modules

# In[ ]:


from math import *
import pandas as pd
import numpy as np

# import plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(['fivethirtyeight'])
mpl.rcParams['lines.linewidth'] = 2
import seaborn as sns

# import the ML algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# pre-processing
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import QuantileTransformer

# import libraries for model validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# import libraries for metrics and reporting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# ### Reading the Dataset and looking at the statistics

# In[ ]:


bookings = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
bookings.head()


# In[ ]:


# Dataset information
bookings.info()


# In[ ]:


#Statistics of the data
bookings.describe().T


# In[ ]:


# Creating a boxplot for Outlier detection
features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'previous_cancellations', 'previous_bookings_not_canceled',
            'days_in_waiting_list', 'adr', 'total_of_special_requests']
n = 1
plt.figure(figsize=(16,18))
for feature in features:
    plt.subplot(3,3,n)
    sns.boxplot(bookings[feature])
    n+=1
    plt.tight_layout()


# ### Statistics shows that there are many Outliers. They will be treated eventually.

# In[ ]:


# Check for the missing data
bookings.isnull().sum()


# In[ ]:


# Some data are missing large in number, and can be conveniently dropped along with unnecessary features
bookings.drop(['agent', 'company', 'arrival_date_week_number'], axis=1, inplace=True)


# In[ ]:


# Lets look into the numbers of children accompanying the adults since there are few missing values in children column
bookings.children.value_counts()


# In[ ]:


# Majority of the visitors were not accompanied by children and hence missing data can be replaced by number of children = 0
bookings.children.fillna(value=0.0, inplace=True)


# In[ ]:


# Iterating the country column by running CountryCoverter revealded that, most of the clients were from Europe. 
# Therefore all missing values are replaced with the country of maximum occurance, Portugal 
bookings.country.fillna(value='PRT', inplace=True)


# ### You may avoid this step , its gonna be hard time for your cpu!

# In[ ]:


pip install country_converter


# In[ ]:


#Lets now convert all the countries to their respective continents to see the continent-wise statistics
import country_converter as coco
cc = coco.CountryConverter()
continents = []
for index, row in bookings.iterrows():
    continent = cc.convert([row.country], to='continent')
    continents.append(continent)
cont_df = pd.DataFrame(continents, columns=['continent'])
bookings = pd.concat([bookings, cont_df], 1)


# In[ ]:


bookings.continent.value_counts()


# In[ ]:


# visualization of continent-wise visitor distribution 
fig, ax = plt.subplots()
plt.axis('equal')
ax.pie(bookings.continent.value_counts(), labels=bookings.continent.value_counts().index, radius=5, autopct='%.2f%%', 
       shadow=True, explode=[1,1,1,1,1,1,1])
plt.show()


# In[ ]:


# Bookings with babies and childres are taken as a single entity 'kids'
bookings['kids'] = bookings.children + bookings.babies
bookings['total_members'] = bookings.kids + bookings.adults


# In[ ]:


# Arrival date to datetime
bookings['arrival_date_year'] = bookings['arrival_date_year'].astype('str')
bookings['arrival_date_month'] = bookings['arrival_date_month'].astype('str')
bookings['arrival_date_day_of_month'] = bookings['arrival_date_day_of_month'].astype('str')
bookings['arrival_date'] = bookings['arrival_date_day_of_month'] + '-' + bookings['arrival_date_month'] + '-' + bookings['arrival_date_year']
bookings['arrival_date'] = pd.to_datetime(bookings['arrival_date'], errors='coerce')


# In[ ]:


# applying string methode to convert to categorical feature
bookings['is_canceled'] = bookings['is_canceled'].astype('str')
bookings['is_repeated_guest'] = bookings['is_repeated_guest'].astype('str')


# In[ ]:


# Missing value visualization 
plt.figure(figsize=(12,7))
sns.heatmap(bookings.isnull(), yticklabels=False)
plt.show()


# ### New dataset confirmed bookings is created

# In[ ]:


confirmed_bookings = bookings[bookings.is_canceled=='0']


# ### Monthly arrivals

# In[ ]:


import datetime as dt
confirmed_bookings['arrival_month'] = bookings['arrival_date'].dt.month
confirmed_bookings.arrival_month.value_counts().sort_index()


# In[ ]:


# Visualization of arrival on monthly basis for both types of hotels together
plt.figure(figsize=(12,5))
(confirmed_bookings.arrival_month.value_counts().sort_index()).plot(kind='bar',
                    figsize=(10,6), title = 'Monthly arrival statistics ', color='#f03b20', alpha=0.5)
plt.xlabel('months')
plt.ylabel('No. of bookings')
plt.xticks(rotation='horizontal')
plt.show()


# ### length of stay preference

# In[ ]:


confirmed_bookings['total_span_of_stay'] = confirmed_bookings.stays_in_week_nights + confirmed_bookings.stays_in_weekend_nights
length_of_stay = confirmed_bookings.total_span_of_stay.value_counts().sort_index()
length_of_stay.head(20).plot(kind='bar', figsize=(10,6), title = 'length of stay', color='#f03b20', alpha=0.5)
plt.title('Stay statistics-Total')
plt.xlabel('length of stay')
plt.ylabel('Bookings')
plt.xticks(rotation='horizontal')
plt.show()


# Just to have an idea of the preferred length of stay for city hotel and resort hotel seperately, we would divide the dataset into two

# In[ ]:


# Making  seperate dataframes for City hotels and Resort hotels
conf_book_city = confirmed_bookings[confirmed_bookings.hotel=='City Hotel']
conf_book_resort = confirmed_bookings[confirmed_bookings.hotel=='Resort Hotel']


# In[ ]:


# Stay statistics based on type of hotel
city = conf_book_city.total_span_of_stay.value_counts().head(20).sort_index()
resort = conf_book_resort.total_span_of_stay.value_counts().head(20).sort_index()

position = list(range(len(city)))
width = 0.25

fig, ax = plt.subplots(figsize=(12,7))
plt.bar([p for p in position], city, width, alpha=0.5, color='#f03b20', label='City')
plt.bar([p+width for p in position], resort, width, alpha=0.5, color='#2c7fb8', label='Resort')

ax.set_xticks([p + width for p in position])
ax.set_xticklabels(city.index)
plt.xlim(min(position)-width, max(position)+width*4)
plt.title('Stay statistics-Seperate')
plt.xlabel('length of stay')
plt.ylabel('Bookings')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()


# ### Weekday statistics of check-in's

# In[ ]:


weekday = bookings.arrival_date.dt.weekday.value_counts().sort_index()

position = list(range(len(weekday))) 
fig, ax = plt.subplots(figsize=(12,6))
plt.bar(position, weekday, width, alpha=0.5, color='#f03b20', label='Cancelled Bookings')

ax.set_xticks([p for p in position])
ax.set_xticklabels(city.index)
ax.set_title('Total bookings for days of the week')
ax.set_xlabel('Bookings')
ax.set_ylabel('Days of the week')
plt.show()


# ### Booking pattern over the years

# In[ ]:


year = bookings.arrival_date.dt.year.value_counts().sort_index()
position = list(range(len(year))) 

fig, ax = plt.subplots(figsize=(12,7))
plt.bar(position, year, width=.50, alpha=0.5, color='#f03b20', label=year.index[0])

ax.set_xticks([p for p in position])
ax.set_xticklabels(year.index)
plt.title('Yearwise bookings')
plt.xlabel('Year')
plt.ylabel('Number of Bookings')
plt.show()


# It is oberved that the year 2017 witnessed most number of check-in's

# ### Evaluating price per night per person

# In[ ]:


# Creating dataframe for price distribution
confirmed_bookings['total_legit_members'] = confirmed_bookings.adults + confirmed_bookings.children
confirmed_bookings_copy = confirmed_bookings.drop(index=confirmed_bookings.loc[confirmed_bookings.total_legit_members==0].index, axis=0)
confirmed_bookings_copy['price_night_person'] = confirmed_bookings_copy.adr/confirmed_bookings_copy.total_legit_members
prices = confirmed_bookings_copy[['hotel', 'reserved_room_type', 'price_night_person']].sort_values('reserved_room_type')
prices.head()


# In[ ]:


# Visualizing the price distribution
plt.figure(figsize=(12,7))
sns.barplot(x=confirmed_bookings_copy.reserved_room_type.sort_values(), y='price_night_person', hue='hotel', 
            data=confirmed_bookings_copy, alpha=0.5, ci='sd', errwidth=2, capsize=0.1)
plt.show()


# ### Room price variation over months

# In[ ]:


# Room price variation
price_variation = confirmed_bookings_copy[['hotel','arrival_month', 'price_night_person']].sort_values('arrival_month')
position = price_variation.arrival_month.value_counts().sort_index().index
fig, ax = plt.subplots(figsize=(12,7))
ax = sns.lineplot(x='arrival_month', y='price_night_person', hue='hotel', data=price_variation, ci='sd')
ax.set_xticks([p for p in position])
ax.set_xticklabels(position)
plt.show()


# ### Treatment of Numerical variables

# In[ ]:


numerical = [var for var in bookings.columns if bookings[var].dtypes!='object']
numerical


# In[ ]:


# after removing irrelevent numerical variables
numerical = ['lead_time',
 'stays_in_weekend_nights',
 'stays_in_week_nights',
 'adults',
 'previous_cancellations',
 'previous_bookings_not_canceled',
 'booking_changes',
 'days_in_waiting_list',
 'adr',
 'required_car_parking_spaces',
 'total_of_special_requests',
 'kids']


# In[ ]:


# Finding the outliers
for j in numerical:
    IQR = bookings[j].quantile(0.75) - bookings[j].quantile(0.25)
    Lower_fence = bookings[j].quantile(0.25) - (IQR * 3)
    Upper_fence = bookings[j].quantile(0.75) + (IQR * 3)
    print(j + ' outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[ ]:


for k in numerical:
    print("the min and max values of " + k + " are {} and {}".format(bookings[k].min(), bookings[k].max()))


# * ### Outliers are : 
# * #### lead_time > 586.0
# * #### stays_in_weekend_nights > 8
# * #### stays_in_week_nights > 9
# * #### adults > 2
# * #### is_repeated_guest > 0
# * #### previous_cancellations > 0
# * #### previous_bookings_not_canceled > 0
# * #### booking_changes > 0
# * #### days_in_waiting_list > 0
# * #### adr > 296.13
# * #### required_car_parking_spaces > 0
# * #### total_of_special_requests > 4
# * #### kids > 0
# * #### total_members > 2

# In[ ]:


def max_value(bookings, variable, top):
    return np.where(bookings[variable]>top, top, bookings[variable])
bookings['lead_time'] = max_value(bookings,'lead_time',586)
bookings['stays_in_weekend_nights'] = max_value(bookings,'stays_in_weekend_nights',8)
bookings['stays_in_week_nights'] = max_value(bookings,'stays_in_week_nights',9)
bookings['adults'] = max_value(bookings,'adults',2)
bookings['previous_cancellations'] = max_value(bookings,'previous_cancellations',0)
bookings['previous_bookings_not_canceled'] = max_value(bookings,'previous_bookings_not_canceled',0)
bookings['booking_changes'] = max_value(bookings,'booking_changes',0)
bookings['days_in_waiting_list'] = max_value(bookings,'days_in_waiting_list',0)
bookings['adr'] = max_value(bookings,'adr',296.13)
bookings['required_car_parking_spaces'] = max_value(bookings,'required_car_parking_spaces',0)
bookings['total_of_special_requests'] = max_value(bookings,'total_of_special_requests',4)
bookings['kids'] = max_value(bookings,'kids',0)   


# ### Treatment of Categorical Variables

# In[ ]:


categorical = [var for var in bookings.columns if bookings[var].dtypes=='object']
categorical


# In[ ]:


# after removing the irrelevent variables
categorical = ['is_canceled',
 'hotel',
 'meal',
 'continent',
 'is_repeated_guest',
 'market_segment',
 'reserved_room_type',
 'assigned_room_type',
 'deposit_type',
 'customer_type',
 'reservation_status']


# In[ ]:


for i in categorical:
    bookings = pd.concat([bookings, pd.get_dummies(bookings[i], drop_first=True)], axis=1)


# ### Features, Labels and train_test_split

# In[ ]:


X = bookings.drop(['is_canceled', 'hotel', 'meal', 'is_repeated_guest', 'market_segment', 'reserved_room_type',
                   'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status','arrival_date_month', 'country',
                  'distribution_channel','reservation_status_date','arrival_date','continent'], axis=1)
y = bookings.is_canceled


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)


# ### Scaling using StandarsScaler

# In[ ]:


#Scaling
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# ###  Classification using Logistic Regression

# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# ### Predictions

# In[ ]:


y_pred = logreg.predict(X_test)


# In[ ]:


confusion_matrix(y_test, y_pred)


# ### Classification metrics

# In[ ]:


print("Accuracy score  : ", accuracy_score(y_test, y_pred))
print("Precision : ", precision_score(y_test, y_pred, pos_label='0'))
print("Recall score : ", recall_score(y_test, y_pred, pos_label='0'))


# ### This was a humble effort of a newbie!
# ### Kindly upvote if you like it!
# ### Suggestions and criticism are welcomed!
