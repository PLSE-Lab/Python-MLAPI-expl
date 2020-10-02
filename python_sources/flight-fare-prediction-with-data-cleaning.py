#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[ ]:


train_data = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.dropna(inplace = True)


# In[ ]:


train_data.isnull().sum()


# Converting into required Formats

# In[ ]:


train_data.drop(['Arrival_Time'],axis=1, inplace=True)


# In[ ]:


# date of journey to date time obj and taking date adn month out
train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month


# In[ ]:


train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[ ]:


train_data.head()


# In[ ]:


# Depature Time
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour

train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute


# In[ ]:


train_data.drop(["Dep_Time"], axis = 1, inplace = True)


# In[ ]:


train_data.head()


# In[ ]:


train_data.drop(['Duration'],axis=1, inplace =True)


# In[ ]:


train_data.head()


# Handling Categorical Data

# In[ ]:


train_data["Airline"].value_counts()


# In[ ]:


# performing oneHotEncoding
Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[ ]:


train_data["Source"].value_counts()


# In[ ]:


#ONeHotEncoding
Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[ ]:


train_data["Destination"].value_counts()


# In[ ]:


# OneHotEncoding
Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[ ]:


train_data["Route"]


# In[ ]:


# most of the "Additional_info" Coloumn is filled with no_info so droping that col and route, Total_stops are correlated to each other so removong one of them
train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[ ]:


train_data["Total_Stops"].value_counts()


# In[ ]:


# LAbelEncoder
train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[ ]:


final_data = pd.concat([train_data, Airline, Source, Destination], axis = 1)


# In[ ]:


final_data.head()


# In[ ]:


final_data.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[ ]:


final_data.head()


# In[ ]:


final_data.shape


# In[ ]:


# features an labels
y=final_data.iloc[:,1]
y.head()


# In[ ]:


final_data.drop(["Price"],axis=1, inplace=True)


# In[ ]:


x=final_data


# In[ ]:


x.head(), x.shape


# In[ ]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


# comparing all regressor models using lazypredict
import lazypredict


# In[ ]:


from lazypredict.Supervised import LazyRegressor
reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
models,predictions = reg.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models


# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor	
reg_rf = HistGradientBoostingRegressor()
reg_rf.fit(X_train, y_train)


# In[ ]:


y_pred = reg_rf.predict(X_test)


# In[ ]:


reg_rf.score(X_train, y_train)


# In[ ]:


reg_rf.score(X_test, y_test)


# In[ ]:


sns.distplot(y_test-y_pred)
plt.show()


# In[ ]:



plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


# calculating RMSE

2090.5509/(max(y)-min(y))


# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


import pickle

file = open('flight_fare_new_model.pkl', 'wb')
pickle.dump(reg_rf, file)


# In[ ]:


model = open('./flight_fare_new_model.pkl','rb')
Hist = pickle.load(model)


# In[ ]:


y_prediction = Hist.predict(X_test)


# In[ ]:


metrics.r2_score(y_test, y_prediction)


# In[ ]:





# In[ ]:




