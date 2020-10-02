#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from math import sqrt, log, exp

import tensorflow as tf

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data= pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


plt.boxplot(data["price"], 0, '')
plt.show()

data["price"].describe()


# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#split in X,y
y=data.loc[:,"price"].apply(lambda x: log(x))

X=data.loc[:,["sqft_living","grade","floors","lat","long","bathrooms","bedrooms","yr_built","yr_renovated","view","waterfront","zipcode","condition","sqft_lot"]]
X["sqft_living"]= X["sqft_living"].apply(lambda x: log(x))
X["lat"]= X["lat"].apply(lambda x: abs(47.63-x))
X["long"]= X["long"].apply(lambda x: log(abs(x)))
X["yr_built"]= X["yr_built"].apply(lambda x: log(abs(x-1955)+1))
X["yr_renovated"]= X["yr_renovated"].apply(lambda x: log(x+1))
X["sqft_lot"]= X["sqft_lot"].apply(lambda x: log(x))

X["lat*long"]=X["lat"]*X["long"]
X["sqft_living*sqft_lot"]=X["sqft_living"]*X["sqft_lot"]

scaler=StandardScaler().fit(X)
X=scaler.transform(X)

# split in train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=123)
#split in validation, test
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=123)


# In[ ]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(16, activation='sigmoid', input_dim=16))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=100, verbose=0)

y_pred = model.predict(X_val)

print("Root Mean Square Error: %2.f" % sqrt(mean_squared_error(np.exp(y_val.values),np.exp(y_pred))))
print('R2 score: %.2f' % r2_score(np.exp(y_val.values),np.exp(y_pred)))


# In[ ]:


# Predicting real life data (Training)
#split in X,y 
y=data.loc[:,"price"].apply(lambda x: log(x))

X=data.loc[:,["sqft_living","sqft_lot","yr_built","bedrooms","bathrooms","lat","long"]]
X["sqft_living"]= X["sqft_living"].apply(lambda x: log(x))
X["sqft_lot"]= X["sqft_lot"].apply(lambda x: log(x))
X["lat"]= X["lat"].apply(lambda x: abs(47.63-x))
X["long"]= X["long"].apply(lambda x: abs(x))
X["yr_built"]= X["yr_built"].apply(lambda x: log(x))


X["lat*long"]=X["lat"]*X["long"]
X["sqft_living*sqft_lot"]=X["sqft_living"]*X["sqft_lot"]

scalerRL=StandardScaler().fit(X)
X=scalerRL.transform(X)

# split in train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)


# In[ ]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

modelRL = Sequential()
modelRL.add(Dense(18, activation='sigmoid', input_dim=9))
modelRL.add(Dense(9, activation='sigmoid'))
modelRL.add(Dense(1, activation='linear'))

modelRL.compile(loss='mse', optimizer='adam')
modelRL.fit(X_train, y_train, epochs=100, verbose=0)

y_pred = modelRL.predict(X_test)

print("Root Mean Square Error: %2.f" % sqrt(mean_squared_error(np.exp(y_test.values),np.exp(y_pred))))
print('R2 score: %.2f' % r2_score(np.exp(y_test.values),np.exp(y_pred)))


# In[ ]:


# Predicting real life data (real examples)

rl_X=pd.DataFrame(columns=["sqft_living","sqft_lot","yr_built","bedrooms","bathrooms","lat","long"])
rl_y=np.empty(10)

## Sold Price
# https://www.zillow.com/homes/recently_sold/King-County-WA/49015189_zpid/207_rid/globalrelevanceex_sort/47.820992,-121.05835,47.042521,-122.534638_rect/9_zm/
rl_X.loc[len(rl_X)]=[1770,7840,1968,4,3,47.7397,-122.185]
rl_y[len(rl_X)-1]=720000
# https://www.zillow.com/homes/recently_sold/King-County-WA/49120416_zpid/207_rid/globalrelevanceex_sort/47.820992,-121.05835,47.042521,-122.534638_rect/9_zm/5_p/
rl_X.loc[len(rl_X)]=[1950,4887,1911,2,2,47.535,-122.388]
rl_y[len(rl_X)-1]=865000
# https://www.zillow.com/homes/recently_sold/King-County-WA/48829308_zpid/207_rid/200000-600000_price/830-2490_mp/globalrelevanceex_sort/47.754559,-121.539002,46.975099,-123.01529_rect/9_zm/
rl_X.loc[len(rl_X)]=[1810,17424,1994,3,3,47.364,-122.043]
rl_y[len(rl_X)-1]=432000
# https://www.zillow.com/homes/recently_sold/King-County-WA/84756911_zpid/207_rid/globalrelevanceex_sort/47.633354,-121.886616,47.439119,-122.255688_rect/11_zm/
rl_X.loc[len(rl_X)]=[1289,1000,2009,2,3,47.532,-122.072]
rl_y[len(rl_X)-1]=520000

i_sale=len(rl_X)

## Sale Price
# https://www.zillow.com/homes/for_sale/King-County-WA/49127321_zpid/207_rid/globalrelevanceex_sort/47.820992,-121.05835,47.042521,-122.534638_rect/9_zm/0_mmm/
rl_X.loc[len(rl_X)]=[2760,4839,1923,4,3,47.557,-122.375]
rl_y[len(rl_X)-1]=550000
# https://www.zillow.com/homes/for_sale/King-County-WA/48662094_zpid/207_rid/200000-600000_price/830-2490_mp/globalrelevanceex_sort/47.804392,-121.289063,47.025674,-122.765351_rect/9_zm/0_mmm/
rl_X.loc[len(rl_X)]=[1290,4791,1925,1,1.5,47.513,-122.387]
rl_y[len(rl_X)-1]=399000
# https://www.zillow.com/homedetails/1102-E-Hemlock-St-Kent-WA-98030/49077132_zpid/
rl_X.loc[len(rl_X)]=[2020,7701,1959,5,2,47.374,-122.220]
rl_y[len(rl_X)-1]=367500
# https://www.zillow.com/homes/for_sale/King-County-WA/48702491_zpid/207_rid/globalrelevanceex_sort/47.633354,-121.886788,47.439118,-122.25586_rect/11_zm/4_p/0_mmm/
rl_X.loc[len(rl_X)]=[2040,8119,1963,4,2,47.502,-122.167]
rl_y[len(rl_X)-1]=435000

rl_X["sqft_living"]= rl_X["sqft_living"].apply(lambda x: log(x))
rl_X["lat"]= rl_X["lat"].apply(lambda x: abs(47.63-x))
rl_X["long"]= rl_X["long"].apply(lambda x: abs(x))
rl_X["yr_built"]= rl_X["yr_built"].apply(lambda x: log(x))
rl_X["sqft_lot"]= rl_X["sqft_lot"].apply(lambda x: log(x))
rl_X["lat*long"]=rl_X["lat"]*rl_X["long"]
rl_X["sqft_living*sqft_lot"]=rl_X["sqft_living"]*rl_X["sqft_lot"]

rl_X=scalerRL.transform(rl_X)

prices_pred= np.exp(modelRL.predict(rl_X))
print("Sold")
for i in range (0,i_sale):
    print("House %i for %i; predicted -> %i" %(i, rl_y[i], prices_pred[i]))
print("\nOn Sale")
for i in range (i_sale,len(rl_X)):
    print("House %i for %i; predicted -> %i" %(i, rl_y[i], prices_pred[i]))

