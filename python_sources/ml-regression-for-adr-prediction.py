#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# import urllib.request
# import re
# import requests
# from bs4 import BeautifulSoup as bs


# For transformations and predictions
# from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.linear_model import LinearRegression
# from scipy.optimize import curve_fit
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor

# For scoring
# from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse

# For validation
from sklearn.model_selection import train_test_split as split


# In[ ]:


df_booking = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
print(df_booking.shape)
df_booking.sample(5)


# In[ ]:


# To find NaN-containing columns:

pd.isnull(df_booking).sum() > 0


# In[ ]:


# Replace NaN in 'children':

df_booking.children.fillna(0, inplace=True)


# In[ ]:


# Create 'total_guests' column: 

df_booking['total_guests']=df_booking['adults'] + df_booking['children'] + df_booking['babies']


# In[ ]:


# Create 'total_nights' column: 

df_booking['total_nights']=df_booking['stays_in_weekend_nights'] + df_booking['stays_in_week_nights']


# In[ ]:


# Replace month name with number:

df_booking['arrival_date_month_number'] = df_booking['arrival_date_month'].replace(['January', 'February', 'March', 'April', 'May', 'June', 'July'
                                                                                  ,'August', 'September', 'October', 'November', 'December']
                                                                                   , [1,2,3,4,5,6,7,8,9,10,11,12]).astype(str).astype(int)


# In[ ]:


# Create binary columns for City/Resort hotel and Portugal/International:

df_booking['hotel_type'] = df_booking['hotel'].replace(['Resort Hotel', 'City Hotel'], [0,1])
df_booking['country_type'] = df_booking['country']
df_booking.loc[(df_booking['country_type'] != 'PRT'), 'country_type'] = 'International'
df_booking['country_type'] = df_booking['country_type'].replace(['International', 'PRT'], [0,1])


# In[ ]:


# Check for outliers in average daily rate (adr) column:

outlier_adr = df_booking.groupby(['adr']).size()
outlier_adr


# In[ ]:


# Consequently removing outliers:

mask= (df_booking['adr']>400) | (df_booking['adr'] <= 0) 
df_booking.loc[mask]
print(df_booking.shape)
df_booking = df_booking.loc[~mask, :]
print(df_booking.shape)


# In[ ]:


# Removing total_guests outliers:

mask = (df_booking['total_guests']>=10) | ((df_booking['adults'] == 0) & (df_booking['children'] == 0)) | (df_booking['babies']>=8) 
print(df_booking.shape)
df_booking = df_booking.loc[~mask, :]
print(df_booking.shape)


# ### ML to Predict ADR will be done using the following models:
# 1. Linear Regression
# 2. Decision Tree

# In[ ]:


# Selecting columns for Regression:

df_booking_full = df_booking[['hotel_type', 'country_type',
                            'arrival_date_month_number', 'stays_in_weekend_nights', 'total_nights',
                            'stays_in_week_nights', 'adults', 'children', 'babies', 'total_guests', 'meal', 
                            'reserved_room_type', 'adr',
                            'total_of_special_requests']].copy()
print(df_booking_full.shape)
df_booking_full.sample(5)


# In[ ]:


# Meals - for Linear Regression will use dummies (see below); for Decision Tree will use scaling:
# SC and Undefined are distinct string values but according to article are the same, therefore combined. 

df_booking_full.loc[(df_booking_full['meal'] == 'SC')| (df_booking_full['meal'] == 'Undefined'), 'meal'] = 'SC_Undefined'

# Scaling - to run before Decision Tree Regression:

# meal_order = ['SC_Undefined', 'BB', 'HB', 'FB']
# meal_map = dict(zip(meal_order, range(len(meal_order))))
# df_booking_full.loc[:, 'meal'] = df_booking_full['meal'].map(meal_map)


# In[ ]:


# Scaling of room types (reserved and assigned):

reserved_room_type_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']
reserved_room_type_map = dict(zip(reserved_room_type_order, range(len(reserved_room_type_order))))

df_booking_full.loc[:, 'reserved_room_type'] = df_booking_full['reserved_room_type'].map(reserved_room_type_map)


# In[ ]:


df_booking_full.sample(5)


# ### df_booking_for_lr (df_booking_full columns adaptation):

# In[ ]:


df_booking_for_lr = df_booking_full[['hotel_type', 'country_type','arrival_date_month_number'
                            , 'stays_in_weekend_nights', 'stays_in_week_nights'
                            , 'adults', 'children', 'babies', 'total_guests', 'total_nights'
                            , 'meal', 'reserved_room_type', 'adr'
                            , 'total_of_special_requests'
                            ]].copy()


# In[ ]:


# Categorizing months into seasons:

def season(x):
    if x in (11, 12, 1, 2, 3):
        x = 'winter'
    elif x in (4, 5, 6, 7, 8):
        x = 'summer'
    else:
        x = 'fall'
    return (x)

df_booking_for_lr['season'] = df_booking_for_lr['arrival_date_month_number'].apply(season)


# In[ ]:


# Categorizing adults, children, babies into is_family:

def is_family(X):
    if ((X.adults > 0) & (X.children > 0)):
        fam = 1
    elif ((X.adults > 0) & (X.babies > 0)):
        fam = 1
    else:
        fam = 0
    return fam

df_booking_for_lr['is_family'] = df_booking_for_lr.apply(is_family, axis = 1)


# In[ ]:


# Categorizing long_stay:

def long_stay(X):
    if (X.total_nights > 7):
        stay = 1
    else:
        stay = 0
    return stay

df_booking_for_lr['long_stay'] = df_booking_for_lr.apply(long_stay, axis = 1)


# In[ ]:


# Categorizing is_weekend:

def is_weekend(X):
    if (X.stays_in_weekend_nights != 0):
        we = 1
    else:
        we = 0
    return we

df_booking_for_lr['is_weekend'] = df_booking_for_lr.apply(is_weekend, axis = 1)


# In[ ]:


# And droping pre-scaling columns:

df_booking_for_lr = df_booking_for_lr.drop(['adults', 'children', 'babies', 'arrival_date_month_number'], axis = 1)

df_booking_for_lr = df_booking_for_lr.drop(['stays_in_weekend_nights', 'stays_in_week_nights'], axis = 1)

# if nights_combination_index is merged, then 'count' should also dropped:
# df_booking_for_lr = df_booking_for_lr.drop(['count'], axis = 1)


# #### Code that was not used:

# In[ ]:


# To create 'stays_in_weekend_nights', 'stays_in_week_nights' combinations, followed by indexing. 
# This will allow us to use the index as measure for number of weekend vs. week nights:

nights = df_booking_for_lr.groupby(['stays_in_weekend_nights', 'stays_in_week_nights']).size().reset_index().rename(columns={0:'count'})
print(nights.shape)
nights['nights_combination_index'] = nights.index.values

# Joining nights DataFrame with df_booking to create a column for the weekend/week nights combinations:

df_booking_for_lr = pd.merge(df_booking_for_lr, nights , on=['stays_in_weekend_nights', 'stays_in_week_nights'])


# In[ ]:


# Linear Regression - 

# 1. Creating dummies for Meals and Season: 

df_booking_for_lr_with_dummies = pd.get_dummies(df_booking_for_lr, drop_first = True)
print(df_booking_for_lr_with_dummies.shape)
df_booking_for_lr_with_dummies.sample(5)


# In[ ]:


# 2. Split:

X = df_booking_for_lr_with_dummies.drop('adr', axis = 1)
y = df_booking_for_lr_with_dummies.adr

X_train, X_test, y_train, y_test = split(X, y, train_size = 0.7, random_state = 142857)


# 3. Fit:

linear_model_1 = LinearRegression().fit(X_train, y_train)


# 4. Predict:

y_train_pred = linear_model_1.predict(X_train)

# 5. Visualize:

ax = sns.scatterplot(x=y_train, y=y_train_pred, color = 'purple' )
ax.plot(y_train, y_train, 'purple')
ax.set_xlabel('y_train')
ax.set_ylabel('y_train_pred')


# In[ ]:


# 6. Inspect:

list(zip(X_train.columns, linear_model_1.coef_))


# In[ ]:


# 7. Score:

linear_model_1_train_mse = np.sqrt(mse(y_train, y_train_pred)).round(2)


# In[ ]:


# 8. Validate:

print('linear_model_1_train_mse is ', linear_model_1_train_mse)

y_test_pred = linear_model_1.predict(X_test)
linear_model_1_test_mse= np.sqrt(mse(y_test, y_test_pred)).round(2)

print('linear_model_1_test_mse is ', linear_model_1_test_mse)


# In[ ]:


# Creating Model Regressor for Hotels and Families: 

class HotelFamilyModel:
    def __init__(self):
        self.city_adults_lm = LinearRegression()
        self.city_families_lm = LinearRegression()
        self.resort_adults_lm = LinearRegression()
        self.resort_families_lm = LinearRegression()


    def fit(self, X, y=None):
        # Fitting the city_adults model
        self.city_adults_lm.fit(X.loc[((X.hotel_type == 1) & (X.is_family == 0)), :], y.loc[((X.hotel_type == 1) & (X.is_family == 0))])
        
        # Fitting the city_families model
        self.city_families_lm.fit(X.loc[((X.hotel_type == 1) & (X.is_family == 1)), :], y.loc[((X.hotel_type == 1) & (X.is_family == 1))])
        
        # Fitting the resort_adults model
        self.resort_adults_lm.fit(X.loc[((X.hotel_type == 0) & (X.is_family == 0)), :], y.loc[((X.hotel_type == 0) & (X.is_family == 0))])
        
        # Fitting the resort_families model
        self.resort_families_lm.fit(X.loc[((X.hotel_type == 0) & (X.is_family == 1)), :], y.loc[((X.hotel_type == 0) & (X.is_family == 1))])
        return self

    def predict(self, X):
        city_adults_df = X.loc[((X.hotel_type == 1) & (X.is_family == 0)), :]
        y_city_adults_pred = pd.Series(self.city_adults_lm.predict(city_adults_df), index=city_adults_df.index)

        city_families_df = X.loc[((X.hotel_type == 1) & (X.is_family == 1)), :]
        y_city_families_pred = pd.Series(self.city_families_lm.predict(city_families_df), index=city_families_df.index)
        
        resort_adults_df = X.loc[((X.hotel_type == 0) & (X.is_family == 0)), :]
        y_resort_adults_pred = pd.Series(self.resort_adults_lm.predict(resort_adults_df), index=resort_adults_df.index)
        
        resort_families_df = X.loc[((X.hotel_type == 0) & (X.is_family == 1)), :]
        y_resort_families_pred = pd.Series(self.resort_families_lm.predict(resort_families_df), index=resort_families_df.index)

        return pd.concat([y_city_adults_pred, y_city_families_pred, y_resort_adults_pred, y_resort_families_pred])


# In[ ]:


hotel_family_model = HotelFamilyModel()

X = df_booking_for_lr_with_dummies.drop('adr', axis = 1)
y = df_booking_for_lr_with_dummies.adr

X_train, X_test, y_train, y_test = split(X, y, train_size = 0.7, random_state = 142857)

hotel_family_model.fit(X_train, y_train)

y_train_pred = hotel_family_model.predict(X_train).reindex(y_train.index)

ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'b')
ax.set_xlabel('y_train')
ax.set_ylabel('y_train_pred')


# In[ ]:


print('linear_model_1_train_rmse is ', linear_model_1_train_mse)
print('linear_model_1_test_rmse is ', linear_model_1_test_mse)
print('hotel_family-model train rmse is ', np.sqrt(mse(y_train, y_train_pred)).round(2))

y_test_pred = hotel_family_model.predict(X_test).reindex(y_test.index)

print('hotel_family-model test rmse is ', np.sqrt(mse(y_test, y_test_pred)).round(2))


# In[ ]:


df_booking_full.loc[(df_booking_full['meal'] == 'SC')| (df_booking_full['meal'] == 'Undefined'), 'meal'] = 'SC_Undefined'
meal_order = ['SC_Undefined', 'BB', 'HB', 'FB']
meal_map = dict(zip(meal_order, range(len(meal_order))))
df_booking_full.loc[:, 'meal'] = df_booking_full['meal'].map(meal_map)


# ### Conclusions for Linear Regression:

# #### Using hotel_family model we get mild improvement of MSE
# #### Predictions using: ADR processing, nights combination index, staying periods did not improve MSE

# ### Decision Tree:
# (Zohar Hirsch contribution)

# In[ ]:


print(df_booking_full.shape)
df_booking_full.loc[(df_booking_full['meal'] == 'SC')| (df_booking_full['meal'] == 'Undefined'), 'meal'] = 'SC_Undefined'
meal_order = ['SC_Undefined', 'BB', 'HB', 'FB']
meal_map = dict(zip(meal_order, range(len(meal_order))))
df_booking_full.loc[:, 'meal'] = df_booking_full['meal'].map(meal_map)


# In[ ]:


df_booking_dt = df_booking_full[['hotel_type', 'country_type'
                            , 'arrival_date_month_number'
                            , 'stays_in_weekend_nights', 'stays_in_week_nights'
                            , 'total_guests'
                            , 'meal', 'reserved_room_type', 'total_of_special_requests'
                            , 'adr'
                            ]].copy()
df_booking_dt.sample(5)


# In[ ]:


# 1. Split:

X = df_booking_dt.drop('adr', axis=1)
y = df_booking_dt.adr

X_train, X_test, y_train, y_test = split(X, y, random_state=312150)

# 2. Assign and Fit:

dt_model = DecisionTreeRegressor(max_leaf_nodes=100)

dt_model.fit(X_train, y_train)


# In[ ]:


get_ipython().system('pip install pydot')
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz


# In[ ]:


# 3. Modeling the Tree:

dot_data = StringIO()  
export_graphviz(dt_model, out_file=dot_data, feature_names=X.columns, leaves_parallel=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
Image(graph.create_png(), width=750) 


# In[ ]:


# Features Importance:

for feature, importance in zip(X.columns, dt_model.feature_importances_):
    print(f'{feature:12}: {importance}')


# In[ ]:


# 4. Predict:

y_train_pred = dt_model.predict(X_train)

# 5. Visualize:

ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')


# In[ ]:


# 6. Score:

RMSE_train = np.sqrt(mse(y_train, y_train_pred)).round(3)

# Validate:

y_test_pred = dt_model.predict(X_test)

RMSE_test = np.sqrt(mse(y_test, y_test_pred)).round(3)

print('Decision Tree train RMSE is ', RMSE_train)
print('Decision Tree test RMSE is ', RMSE_test)


# ## Decision Tree predicts ADR better than Linear Models
