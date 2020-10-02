#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


# Read the training data

# In[ ]:


df= pd.read_excel(r'/kaggle/input/flight-fare-prediction-mh/Data_Train.xlsx')


# # Data Analysis

# In[ ]:


df.head(20)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Airline'].value_counts()


# In[ ]:


df["Source"].value_counts()


# In[ ]:


df["Destination"].value_counts()


# In[ ]:


df["Route"].value_counts()


# In[ ]:


df['Duration'].value_counts()


# In[ ]:


df["Total_Stops"].value_counts()


# # Data Cleaning

# Converting day and month to different coloums using pandas datetime function and removing Date_of_journey

# In[ ]:


df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format = "%d/%m/%Y").dt.month
df.drop(["Date_of_Journey"], axis = 1, inplace = True)


# #converting arrival/ depature hour and minutes to different coloums using pandas datetime function and removing Arrival_time/Dep_time
# 

# In[ ]:


df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
df.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[ ]:


df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
df.drop(["Dep_Time"], axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


# Assigning and converting Duration column into list
duration = list(df["Duration"])

for i in range(len(duration)):
     #Check if duration contains only hour or mins
    if len(duration[i].split()) != 2:   
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   
        else:
            duration[i] = "0h " + duration[i]           

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extracting hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracting minutes from duration
    
df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins

df.drop(["Duration"], axis = 1, inplace = True)


# In[ ]:


df.head()


# OneHotEncoding

# In[ ]:


Airline = df[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first= True)
Airline.head()


# In[ ]:


Source = df[["Source"]]
Source = pd.get_dummies(Source, drop_first= True)
Source.head()


# In[ ]:


Destination = df[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first = True)
Destination.head()


# Since routes and total stops are related we drop routes.
# Additional info in this case is not of much use.

# In[ ]:


df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[ ]:


df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
df.head()


# In[ ]:


data = pd.concat([df, Airline, Source, Destination], axis = 1)
data.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# Cleaned dataset ready to fit in the model.

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# # Splitting to training and validation set.

# In[ ]:


X = data.drop(columns='Price')
y = data.iloc[:, 1]


# Applying Random Forest Regressor.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = RandomForestRegressor()
model.fit(X_train, y_train)


# Results.

# In[ ]:


y_pred = model.predict(X_test)
print("Train Accuracy: ", model.score(X_train, y_train))
print("Test Accuracy: ", model.score(X_test, y_test))


# Using RandomizedSearchCV to obtain optimized parameters inorder yo improve accuracy.

# In[ ]:


random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(5, 30, num = 6)],
               'min_samples_split': [2, 5, 10, 15, 100],
               'min_samples_leaf':  [1, 2, 5, 10]}

rf_rand = RandomizedSearchCV(estimator = model, param_distributions = random_grid,scoring='neg_mean_squared_error',
                               n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_rand.fit(X_train,y_train)


# In[ ]:


rf_rand.best_params_


# Model with optimized parameters.

# In[ ]:


model = RandomForestRegressor(n_estimators= 700, min_samples_split = 15,
                              min_samples_leaf= 1, max_features = 'auto',
                              max_depth= 20)


# In[ ]:


model.fit(X_train, y_train)
prediction = model.predict(X_test)


# # Final Results

# In[ ]:


print("Train Accuracy: ", model.score(X_train, y_train))
print("Test Accuracy: ", model.score(X_test, y_test))


# Saving the model.

# In[ ]:


import pickle
file = 'flight_model.pkl'
pickle.dump(model, open(file, 'wb'))


# # Conclusion
# 
# We have successfully obtained a model with Test Accuracy of 81.2%.
# 
# 
# # ThankYou!!
