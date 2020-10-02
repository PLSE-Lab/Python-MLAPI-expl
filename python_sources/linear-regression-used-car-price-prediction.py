#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression
# Over 370,000 used cars were scraped from Ebay-Kleinanzeigen. The content of the data is in German. The data is available [here](https://www.kaggle.com/orgesleka/used-cars-database) The fields included in the file data/autos.csv are:
# 
# * seller : private or dealer
# * offerType
# * vehicleType
# * yearOfRegistration : at which year the car was first registered
# * gearbox
# * powerPS : power of the car in PS
# * model
# * kilometer : how many kilometers the car has driven
# * monthOfRegistration : at which month the car was first registered
# * fuelType
# * brand
# * notRepairedDamage : if the car has a damage which is not repaired yet
# * price : the price on the ad to sell the car.
#  
#  #### Goal
# #### Given the characteristics/features of the car, the sale price of the car is to be predicted.
# 
# 

# In[17]:


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


# In[18]:


#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns


# In[19]:


cars = pd.read_csv("../input/autos.csv",encoding='latin1')


# In[20]:


cars.head()


# <h3>Checking for nulls in all the columns<h3>

# In[21]:


cars.isnull().sum()


# <h3>Dropping all the rows which have null values</h3>

# In[22]:


cars_updated = cars.dropna()


# <h4>Making sure there are no null values present<h4>

# In[23]:


cars_updated.isnull().sum()


# <h3>Cleaning the data</h3>
# <h4>Dropping the columns abtest,noofpictures,offertype, datecrawled,Seller,name<h4>
# * Reason: abtest,noofpictures values are same for all
# * offertype :  as there are only two types Angepbot(Offer) which is the majority and Guesh(Request) is only handful so this would not effect the output
# *datecrawled : just gives when the system got the data
# * Seller : has two types private and commercial which willnot effect the price it doesnot matter who sells the vehicle
# * name : as the columns model and brand already has this info
# * postalcode : as we cannot directly determine it based on postal code

# In[24]:


cars_updated = cars_updated.iloc[:,[6,7,8,9,10,11,12,13,14,15,4]]


# In[25]:


cars_updated.columns


# <h3>Checking for the Unique values in the columns</h3>

# In[26]:


print('Vehicle Type: ',cars_updated.vehicleType.unique())
print('Gearbox: ',cars_updated.gearbox.unique())
print('Fuel Type: ',cars_updated.fuelType.unique())
print('Repaired Damage: ',cars_updated.notRepairedDamage.unique())


# <h3>As the data is in German we need to change the Column values to English</h3>

# In[27]:


cars_updated.replace({'gearbox':{'manuell':'manual','automatik':'automatic'}},inplace=True)
cars_updated.replace({'vehicleType':{'kleinwagen':'small_car','kombi':'combi','andere':'Others'}},inplace=True)
cars_updated.replace({'fuelType':{'benzin':'petrol','andere':'others','elektro':'electro'}},inplace=True)
cars_updated.replace({'notRepairedDamage':{'nein':'no','ja':'yes'}},inplace=True)


# In[28]:


cars_updated.head(10)


# <h3>Cleaning the columns Price, Year of Registration,  Power & Kilometer</h3>

# In[29]:


cars_updated = cars_updated.loc[(cars_updated.price>400)&(cars_updated.price<=40000)]
cars_updated = cars_updated.loc[(cars_updated.yearOfRegistration>1990)&(cars_updated.yearOfRegistration<=2016)]
cars_updated = cars_updated.loc[(cars_updated.powerPS>10)]
cars_updated = cars_updated.loc[(cars_updated.kilometer>1000)&(cars_updated.kilometer<=150000)]


# <h3>Adding a column</h3>
# Creating a field **Days_old** from columns **Registration_Year and Registration_Month**

# In[30]:


#Replacing all the 0 month values to 1
cars_updated.monthOfRegistration.replace(0,1,inplace=True)
# Making the year and month column to get a single date
Purchase_Datetime=pd.to_datetime(cars_updated.yearOfRegistration*10000+cars_updated.monthOfRegistration*100+1,format='%Y%m%d')
import time
from datetime import date
y=date(2018, 5,1)
# Calculating days old by subracting both date fields and converting them into integer
Days_old=(y-Purchase_Datetime)
Days_old=(Days_old / np.timedelta64(1, 'D')).astype(int)
#type(Days_old[1])
cars_updated['Days_old']=Days_old


# <h3>Dropping Year of Registration and Month of Registration as we have Days_Old Column.</h3>

# In[31]:


cars_updated.drop(columns=['yearOfRegistration','monthOfRegistration','powerPS'],inplace=True)


# <h3>Creating dummies for Categorical Columns - Fuel Type, Not repaired damage, Vehicle Type, Model, Brand</h3>

# In[32]:


cars_dummies=pd.get_dummies(data=cars_updated,columns=['notRepairedDamage','vehicleType','model','brand','gearbox','fuelType'])


# In[33]:


cars_dummies.head(10)


# In[34]:


X = cars_dummies.drop('price',axis=1)


# In[35]:


y = cars_dummies.price


# In[36]:


X.head(5)


# In[37]:


y.head(5)


# In[38]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)


# In[39]:


print (linreg.intercept_)


# In[40]:


# pair the feature names with the coefficients
list(zip(X.columns.get_values(), linreg.coef_))


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=123)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)


# In[42]:


np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# In[43]:


#Predicting the test set results
y_pred = linreg.predict(X_test)
print(linreg.score(X_test, y_test)*100,'% Prediction Accuracy')

