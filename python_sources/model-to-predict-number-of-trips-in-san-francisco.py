#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


import statistics


# In[ ]:


from datetime import datetime


# In[ ]:


df = pd.read_csv("../input/trip.csv")


# In[ ]:


df.head()


# In[ ]:


#Convert to datetime so that it can be manipulated more easily
df.start_date = pd.to_datetime(df.start_date, format='%m/%d/%Y %H:%M')


# In[ ]:


#Extract the year, month, and day from start_date
df['date'] = df.start_date.dt.date


# In[ ]:


#Each entry in the date feature is a trip. 
#By finding the total number of times a date is listed, we know how many trips were taken on that date.
dates = {}
for d in df.date:
    if d not in dates:
        dates[d] = 1
    else:
        dates[d] += 1


# In[ ]:


#Create the data frame that will be used for training, with the dictionary we just created.
df2 = pd.DataFrame.from_dict(dates, orient = "index")


# In[ ]:


df2['date'] = df2.index


# In[ ]:


df2['trips'] = df2.iloc[:,0]


# In[ ]:


df2.head()


# In[ ]:


train = pd.DataFrame(df2.date)


# In[ ]:


train['trips'] = df2['trips']


# In[ ]:


train.head()


# In[ ]:


train.reset_index(drop = True, inplace = True)


# In[ ]:


train.head()


# In[ ]:


train = train.sort_values(by='date')


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


type(train.date[0])


# In[ ]:


weather = pd.read_csv("../input/weather.csv")


# In[ ]:


weather.head()


# In[ ]:


weather.events.unique()


# In[ ]:


weather.loc[weather["events"] == 'rain', 'events'] = 'Rain'


# In[ ]:


weather.events.unique()


# In[ ]:


weather.loc[weather["events"].isnull(), 'events'] = 'Normal'


# In[ ]:


weather.events.unique()


# Checking the zip code based data:

# In[ ]:


weather.zip_code.unique()


# In[ ]:


for zipcode in (weather.zip_code.unique()):
    print(zipcode)
    print(weather[weather.zip_code == zipcode].isnull().sum())
    print()


# Data is clean for 94107, 95113
# Choosing 94107 - San Francisco as more work carried on for this zip code and it is easier to check.

# In[ ]:


weather = weather[weather.zip_code == 94107]


# In[ ]:


weather = weather.drop(['zip_code'], axis=1)


# to fill nulls for max_gust speed

# In[ ]:


weather.max_gust_speed_mph.describe()


# In[ ]:


weather.corr()


# In[ ]:


w1 = weather.loc[:, ('max_wind_Speed_mph', 'max_gust_speed_mph')]


# max_wind_Speed_mph and max_gust_speed_mph are correlated

# In[ ]:


w1.corr()


# In[ ]:


w1_null = w1[w1.max_gust_speed_mph.isnull()]


# In[ ]:


w1_null.head()


# In[ ]:


weather.loc[weather.max_gust_speed_mph.isnull(), 'max_gust_speed_mph'] = weather.max_wind_Speed_mph


# In[ ]:


weather.max_gust_speed_mph.isnull().sum()


# In[ ]:


weather.iloc[63]


# In[ ]:


for i in weather.precipitation_inches[0:5]:
    print(type(i))


# In[ ]:


weather.precipitation_inches = pd.to_numeric(weather.precipitation_inches, errors = 'coerce')


# In[ ]:


type(weather.precipitation_inches.iloc[1])


# In[ ]:


weather.precipitation_inches.describe()


# In[ ]:


statistics.median(weather[weather.precipitation_inches.notnull()].precipitation_inches)


# In[ ]:


weather.precipitation_inches.isnull().sum()


# In[ ]:


weather.loc[weather.precipitation_inches.isnull(), 'precipitation_inches'] = 0.0


# In[ ]:


weather.precipitation_inches.isnull().sum()


# In[ ]:


weather = weather.sort_values(by = 'date')


# In[ ]:


weather.reset_index(drop = True, inplace = True)


# In[ ]:


weather.date.head()


# Merging weather to train

# In[ ]:


train = train.merge(weather, on = train.date)


# In[ ]:


train.head()


# In[ ]:


train.drop(['key_0', 'date_y'],1, inplace= True)


# In[ ]:


train = train.rename(columns={'date_x':'date'})


# In[ ]:


train.head()


# In[ ]:


stations = pd.read_csv("../input/station.csv")


# In[ ]:


stations.head()


# In[ ]:


stations.city.unique()


# In[ ]:


stations = stations[stations.city == 'San Francisco']


# In[ ]:


stations.reset_index(drop = True, inplace = True)


# In[ ]:


stations.shape


# In[ ]:


stations.head()


# In[ ]:


for i in stations.installation_date[0:5]:
    print(i, type(i))


# In[ ]:


stations.installation_date.shape


# In[ ]:


stations.installation_date = pd.to_datetime(stations.installation_date)


# In[ ]:


stations['installation_date'] = stations.installation_date.dt.date


# In[ ]:


for str in stations.installation_date[0:5]:
    print(type(str))


# In[ ]:


print (stations.installation_date.min())
print (stations.installation_date.max())


# Dock installations have been happening during this period.

# In[ ]:


#For each day in train.date, find the number of docks (parking spots for individual bikes) that were installed 
#on or before that day.
total_docks = []
for day in train.date:
    total_docks.append(sum(stations[stations.installation_date <= day].dock_count))


# In[ ]:


train['total_docks'] = total_docks


# In[ ]:


train.total_docks.unique()


# Holidays

# In[ ]:


from pandas.tseries.holiday import USFederalHolidayCalendar


# In[ ]:


#Find all of the holidays during out time span
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=train.date.min(), end=train.date.max())


# In[ ]:


holidays


# In[ ]:


from pandas.tseries.offsets import CustomBusinessDay


# In[ ]:


#Find all of the business days in our time span
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
business_days = pd.DatetimeIndex(start=train.date.min(), end=train.date.max(), freq=us_bd)


# In[ ]:


business_days


# In[ ]:


business_days = pd.to_datetime(business_days, format = '%Y/%m/%d').date


# In[ ]:


# if train.date is a business day or not
train['business_days'] = train.date.isin(business_days)


# In[ ]:


train['business_days'].head()


# In[ ]:


holidays = pd.to_datetime(holidays, format = '%Y/%m/%d').date


# In[ ]:


# if train.date is a holiday or not
train['holidays'] = train.date.isin(holidays)


# In[ ]:


train['holidays'].head()


# In[ ]:


weekday = []
for i in train.date:
    wkday = i.weekday()
#    print(wkday)
    if wkday in range(0,5):
        weekday.append(1)
#        print(1)
    else:
        weekday.append(0)
#        print(0)


# In[ ]:


train['weekday'] = weekday


# In[ ]:


train.head()


# In[ ]:


train.business_days = [1 if i is True else 0 for i in train.business_days ]


# In[ ]:


train.holidays = [1 if i is True else 0 for i in train.holidays ]


# In[ ]:


train.head()


# In[ ]:


train['month'] = pd.to_datetime(train.date).dt.month


# In[ ]:


train.head()


# In[ ]:


labels = train.trips


# In[ ]:


train.drop(['date', 'trips'],1, inplace = True)


# In[ ]:


train.tail()


# Train the model

# In[ ]:


events = pd.get_dummies(train.events, drop_first = True)


# In[ ]:


train = train.merge(events, left_index = True, right_index = True)


# In[ ]:


train.head()


# In[ ]:


train.drop(['events'], axis = 1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state = 1)


# In[ ]:


import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# In[ ]:


regressor = LinearRegression()


# In[ ]:


predicted = cross_val_predict(regressor, X_train, y_train, cv=15)


# In[ ]:


import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.scatter(y_train, predicted, edgecolors = (0,0,0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(regressor, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   


# Does not provide good prediction with Linear model

# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rfr = RandomForestRegressor(n_estimators = 55,
                            min_samples_leaf = 3,
                            random_state = 2, bootstrap=False)


# In[ ]:


predicted = cross_val_predict(rfr, X_train, y_train, cv=15)


# In[ ]:


fig,ax = plt.subplots()
ax.scatter(y_train, predicted, edgecolors = (0,0,0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(rfr, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   


# In[ ]:


rfr1 = RandomForestRegressor(n_estimators=60, criterion='mse', random_state=2)


# In[ ]:


predicted = cross_val_predict(rfr1, X_train, y_train, cv=15)


# In[ ]:


fig,ax = plt.subplots()
ax.scatter(y_train, predicted, edgecolors = (0,0,0))
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(rfr1, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   


# Knn Regressor:

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train) 


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(neigh, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   


# In[ ]:


neigh1 = KNeighborsRegressor(n_neighbors=3)
neigh1.fit(X_train, y_train) 


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(neigh1, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   


# Knn Regressor not predicting well

# Gradient Boosting Regressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


gbr = GradientBoostingRegressor(learning_rate = 0.12,
                                n_estimators = 150,
                                max_depth = 8,
                                min_samples_leaf = 1,
                                random_state = 2)


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(gbr, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))   


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


dtr = DecisionTreeRegressor(min_samples_leaf = 3,
                            max_depth = 8,
                            random_state = 2)


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(dtr, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


abr = AdaBoostRegressor(n_estimators = 100,
                        learning_rate = 0.1,
                        loss = 'linear',
                        random_state = 2)


# In[ ]:


scoring = ['r2','neg_mean_squared_error','neg_mean_absolute_error']
for i in scoring:
    scores = cross_val_score(abr, X_train, y_train, cv=15, scoring = i)
#    print(scores)
    if i == 'r2':
        print(i, ': ', scores.mean())
    elif i == 'neg_mean_squared_error':    
        x = -1*scores.mean()
        y = math.sqrt(x) 
        print('RMSE: ', "%0.2f" % y)
    elif i == 'neg_mean_absolute_error':
        x = -1*scores.mean()
        print(i, ": %0.2f (+/- %0.2f)" % (x, scores.std() * 2))     


# # High r2_score models:
# rfr1 - r2: 0.842963044947498, RMSE: 155.81, neg_mean_absolute_error : 105.08 (+/- 28.14)              
# abr  - r2: 0.8029860904336908, RMSE: 174.65, neg_mean_absolute_error : 124.49 (+/- 31.38)             
# gbr  - r2: 0.7993544716660546, RMSE: 176.35, neg_mean_absolute_error : 115.88 (+/- 29.64)             
# rfr  - r2: 0.7615315265512614, RMSE: 191.96, neg_mean_absolute_error : 132.13 (+/- 27.82)

# The best model is rfr1.
# Predicting the number of trips with this model.

# In[ ]:


rfr1.fit(X_train, y_train )
predicted = rfr1.predict(X_test)


# In[ ]:


labels.describe()


# In[ ]:


y_test.reset_index(drop = True, inplace = True)


# In[ ]:


plt.figure(figsize=(10,7))
plt.plot(predicted)
plt.plot(y_test)
plt.legend(['Prediction', 'Acutal'])
plt.ylabel("Number of Trips", fontsize = 14)
plt.xlabel("Predicted Date", fontsize = 14)
plt.title("Predicted Values vs Actual Values", fontsize = 14)
plt.show()

