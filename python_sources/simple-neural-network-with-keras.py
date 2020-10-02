#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/bike-sharing-dataset"))

# Any results you write to the current directory are saved as output.


# In[ ]:


raw = pd.read_csv("../input/bike-sharing-dataset/hour.csv")


# ## Now, we are going to explore that data and understand it. The description reads as this
# 
# Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
# 	
# 	- instant: record index
# 	- dteday : date
# 	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
# 	- yr : year (0: 2011, 1:2012)
# 	- mnth : month ( 1 to 12)
# 	- hr : hour (0 to 23)
# 	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
# 	- weekday : day of the week
# 	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# 	+ weathersit : 
# 		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# 	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
# 	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
# 	- hum: Normalized humidity. The values are divided to 100 (max)
# 	- windspeed: Normalized wind speed. The values are divided to 67 (max)
# 	- casual: count of casual users
# 	- registered: count of registered users
# 	- cnt: count of total rental bikes including both casual and registered

# In[ ]:


raw.head()


# ## Lets get a deeper look

# In[ ]:


raw.describe()


# ##  Lets check the categorical variables now.
# 
# ### We have some variables such as the week days in which we do NOT really want to use numbers, but we just simply want to denotate whether or not a bicycle was used in a given day (Monday, Tuesday). At the moment that is done by assigning to the column "weekday" a value between 0 and 6, we want to change that... lets use dummy variables

# In[ ]:


def generate_dummies(df, dummy_column):
    dummies = pd.get_dummies(df[dummy_column], prefix=dummy_column)
    df = pd.concat([df, dummies], axis=1)
    return df

X = pd.DataFrame.copy(raw)
dummy_columns = ["season",     # season (1:springer, 2:summer, 3:fall, 4:winter)
                 "yr",          # year (0: 2011, 1:2012)
                 "mnth",        # month ( 1 to 12)
                 "hr",          # hour (0 to 23)
                 "weekday",     # weekday : day of the week
                 "weathersit"   # weathersit : 
                                 # - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
                                 # - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
                                 # - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
                                 # - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
                ]
for dummy_column in dummy_columns:
    X = generate_dummies(X, dummy_column)


# In[ ]:


X.head()


# In[ ]:


X.columns


# ## Now we need to drop the columns used originally for dummies, notice that now we have weekday_0, weekday_1 ... weekday_6, which represents Sunday to Monday (personal note here!!: I am Spanish and in Spain weekday 0 would be Monday... in English however the first day of the week is Sunday... keep in in mind!)
# 
# ### In any case, despite having weekday_1... weekday_6 we still have the column weekday, which is of no use already, so lets remove it along with the rest of dummy columns

# In[ ]:


for dummy_column in dummy_columns:
    del X[dummy_column]

X.columns


# ### And now, lets see how our data looks like

# In[ ]:


X.head()


# In[ ]:


X.describe()


# ### Time for us to plot some data and get an idea of what's going on here

# In[ ]:


first_3_weeks = 3*7*24 # 3 weeks (7 days), 24 hours each day
X[:first_3_weeks].plot(x='dteday', y='cnt', figsize=(18, 5))


# ### It is also obvious that we do not need the "instant", "'dteday" columns, lets remove them

# In[ ]:


del X["instant"]
del X["dteday"]


# ### Finally, we need to declare which one will be our "target" column, that is, what do we want to predict? in this case it would be either "casual", "registered" or "cnt". I will use "cnt"

# In[ ]:


y = X["cnt"]
del X["cnt"]
del X["registered"]
del X["casual"]


# In[ ]:


X.head()


# ## We will now split into train data and test data, using 70% as train data

# In[ ]:


all_days = len(X) // 24
print("Total observations", len(X))
print("Total number of days", all_days)
days_for_training = int(all_days * 0.7)
X_train = X[0:days_for_training]
X_test = X[days_for_training:]


# In[ ]:


print("Observations for training", len(X_train))
print("Observations for testing", len(X_test))
print("Some target values", y.head())


# ### We still need to normalize our target values!

# In[ ]:


y_normalized = (y - y.min()) / (y.max() - y.min())
y_normalized.head()

y_train = y[0:days_for_training]
y_test = y[days_for_training:]
y_train_normalized = y_normalized[0:days_for_training]
y_test_normalized = y_normalized[days_for_training:]


# ## We will now build a simple model

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
features = X.shape[1]
model = Sequential()
model.add(Dense(13, input_shape=(features,), activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(1, activation='linear'))

model.summary()


# In[ ]:


from keras.optimizers import SGD
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss="mean_squared_error")


# In[ ]:


results = model.fit(X_train, y_train_normalized, epochs=10, validation_data = (X_test, y_test_normalized))


# In[ ]:


results.history
pd.DataFrame.from_dict(results.history).plot()


# In[ ]:




