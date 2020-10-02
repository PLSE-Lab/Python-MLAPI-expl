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

# Any results you write to the current directory are saved as output.


# In[ ]:


import json
import pandas as pd
import numpy as np
import os


# In[ ]:


#Load the Amsterdamjson file
ams = pd.read_json('../input/amsterdam.json')


# In[ ]:


ams.head()


# In[ ]:


ams.shape


# In[ ]:


ams.columns


# In[ ]:


ams.isnull().sum()


# In[ ]:


ams.describe()


# In[ ]:


#Print head of the Price column (Target variable)
ams['price'].head()


# In[ ]:


#Remove the '$' and ',' symbols
ams['price']=ams['price'].str.replace(',','')
ams['price']=ams['price'].str.replace('$','')


# In[ ]:


#Check if the symbols were removed
ams['price'].head()


# In[ ]:


type(ams['price'])


# In[ ]:


ams=pd.DataFrame(ams)


# In[ ]:


ams=ams.convert_objects(convert_numeric=True)


# In[ ]:


ams.describe()


# In[ ]:


dt=pd.DataFrame(ams['price'])


# In[ ]:


ams.dtypes


# In[ ]:


#Drop all places whose price is less than 440
ams=ams[(ams.price<=440.0)]
ams=ams.dropna()


# In[ ]:


a_latitude=52.379189
a_longitude=4.899431


# In[ ]:


#Find the distance of all the observations in the data from the city centre in km units. 
#Assume that the city centre has a latitude = 52.379189 and longitude = 4.899431.
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km


for index, row in ams.iterrows():
    ams.loc[index, 'distance'] = haversine(a_longitude, a_latitude, row['longitude'], row['latitude'])


# In[ ]:


ams.head()


# In[ ]:


#Filter room type = Private room and distance < 1
ams=ams.convert_objects(convert_numeric=True)


dt3=ams[ams.room_type=='Private room']
dt3=dt3[dt3.distance<=1]
dt3.price.describe()


# In[ ]:


#Create dummies
ams=pd.get_dummies(ams)


# In[ ]:


ams.head()


# In[ ]:


#Assign predictor matrix to X
X=ams.drop(['price'],axis=1)


# In[ ]:


#Assign Target matrix to Y
Y=ams['price']


# In[ ]:


X.head()


# In[ ]:


Y.head()


# In[ ]:


#Split into Train and Test
import sklearn.model_selection as model_selection

X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=0.3,random_state=400)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#Tuning the n_estimators
for i in range(30,610,10):
    reg=RandomForestRegressor(n_estimators=i,max_depth=5,max_features='sqrt',oob_score=True,random_state=200)
    reg.fit(X_train,Y_train)
    oob=reg.oob_score_
    print('For n_estimators = '+str(i))
    print('OOB score is '+str(oob))
    print('************************')


# In[ ]:


from sklearn import metrics
metrics.mean_squared_error(Y_test,reg.predict(X_test))


# In[ ]:


#Run a Random Forest regressor for n=330 which gives us the maximum OOB score
reg=RandomForestRegressor(n_estimators=330,max_depth=5,max_features='sqrt',oob_score=True)
reg.fit(X_train,Y_train)


# In[ ]:


reg.score(X_test,Y_test)


# In[ ]:


reg.oob_score_


# In[ ]:


#Check for feature importances
reg.feature_importances_


# In[ ]:


#Sort the feature importance vlues ina descending order to get the most important features 
imp_feat=pd.Series(reg.feature_importances_,index=X.columns.tolist())


# In[ ]:


#Print the important feature in descending order
imp_feat.sort_values(ascending=False)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#Validate the model on test data
pred=reg.predict(X_test)
pred


# In[ ]:


Y=pd.DataFrame(ams['price'])


# In[ ]:


#Find Errors by calculating the difference between the actual and rpedicted values
Errors=Y_test-pred


# In[ ]:


Errors.head()


# In[ ]:


#Scatter plot between the Actual values and Error term
plt.scatter(Y_test, Errors, alpha=0.5)


# In[ ]:




