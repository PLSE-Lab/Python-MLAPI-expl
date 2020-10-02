#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection # for splitting the data into training and testing data
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


train_data = pd.read_csv("../input/autos.csv" ,encoding = "ISO-8859-1" )


# In[8]:


print(train_data.shape)


# *Let's understand the data in deep*

# In[9]:


train_data.head()


# In[10]:


train_data.describe()


# In[11]:


train_data.isnull().sum()


# 5 columns have nan values

# ## Look at the seller column

# In[12]:


train_data["seller"].value_counts()


# Only 3 belong to second seller , this is interesting and also we get to know this feature has not of any use

# In[4]:


del train_data["seller"]


# ## Look at  the offerType column

# In[16]:


train_data["offerType"].value_counts()


# Again second offerType has only 12 entries , we should remove this column 

# In[5]:


del train_data["offerType"]


# In[19]:


train_data.head(3) 


# **Look at the nrOfPictures column**

# In[21]:


train_data["nrOfPictures"].value_counts()


# this column only take one value i.e., 0 , so i think there is no advantage of consider this column in our dataset

# In[6]:


del train_data["nrOfPictures"]


# ## Look at the abtest column

# In[25]:


train_data["abtest"].value_counts()


# Both values are popular , we should keep this column

# ## Let's inspect the date features

# In[30]:


train_data[ ["dateCrawled","dateCreated","lastSeen"] ].head()


# These dates basically represent when was the ads crawled , or time of lastseen of these ads , we will not able to collect much information from these dates so it's better to remove them

# In[7]:


train_data = train_data.drop(["dateCrawled","dateCreated","lastSeen"] , axis=1 )


# In[32]:


train_data.head()


# ## Come to the name column

# In[33]:


train_data["name"].head()


# name only represent the name of the car which is useless here

# In[8]:


del train_data["name"]


# In[35]:


train_data.isnull().sum()


# ## Now It's time to fill the NaN in columns 
# ## 1) gearbox
# ## 2) notRepairedDamage
# ## 3) fuelType
# ## 4) VehicleType
# ## 5) Model

# In[36]:


train_data["gearbox"].value_counts()


# There are two types of gearbox but frequency of manuell is higher than automatik but we cannot fill nan values with maxFreq value , so let's see how can we fill it

# In[38]:


# we can see brand column has no nans
train_data["brand"].isnull().sum()


# In[39]:


#so we will use brand column to fill gearbox values
train_data.groupby("brand")["gearbox"].value_counts()


# In[9]:


gearbox = train_data["gearbox"].unique()
brand = train_data["brand"].unique()
d = {}

for i in brand :
    m = 0
    for j in gearbox :
        if train_data[(train_data.gearbox == j) & (train_data.brand == i)].shape[0] > m :
            m = train_data[(train_data.gearbox == j) & (train_data.brand == i)].shape[0]
            d[i] = j
        


# In[11]:


for i in brand :
    train_data.loc[(train_data.brand == i) & (train_data.gearbox.isnull()) ,"gearbox" ] = d[i]


# In[12]:


# no nans in gearbox
train_data["gearbox"].isnull().sum()


# ## let's look at the notRepairedDamage

# In[14]:


train_data["notRepairedDamage"].value_counts()


# In[13]:


train_data["notRepairedDamage"].isnull().sum()


# 72060 values are nans and we can see 'nein' value is more frequent than 'ja' , so we can fillna with maxFreq value bcz this column is not much indicating much importance

# In[15]:


train_data["notRepairedDamage"].fillna("nein",inplace = True)


# In[16]:


train_data["notRepairedDamage"].isnull().sum()


# **Let's handle the NaN in FuelType**

# In[18]:


train_data["fuelType"].value_counts()


# we can see only benzin and diesel are more frequent  ,  for now lets fill it with 'benzin'

# In[21]:


train_data["fuelType"].fillna("benzin",inplace = True)


# In[22]:


train_data.isnull().sum()


# **now fill the vehicleType nan**

# In[23]:


train_data["vehicleType"].value_counts()


# here no particular value is more frequent , all top 3 types are frequent

# In[25]:


# we can fill according to the fuelType values
train_data.groupby("fuelType")["vehicleType"].value_counts()


# here we see different fuelType correspond to different vehicleType

# In[33]:


vehicleType = train_data["vehicleType"].unique()
fuelType = train_data["fuelType"].unique()
print(fuelType)
print(vehicleType)
#remove nan 
vehicleType = np.delete(vehicleType,0)


# In[34]:


d = {}
for i in fuelType :
    m = 0
    for j in vehicleType :
        if train_data[(train_data.vehicleType == j) & (train_data.fuelType == i)].shape[0] > m :
            m = train_data[(train_data.vehicleType == j) & (train_data.fuelType == i)].shape[0]
            d[i] = j


# In[35]:


for i in fuelType :
    train_data.loc[(train_data.fuelType == i) & (train_data.vehicleType.isnull()) ,"vehicleType" ] = d[i]


# In[36]:


train_data["vehicleType"].isnull().sum()


# **Lets fill the model**

# In[39]:


len(train_data["model"].unique())


# Too much unique values in model type lets find the maximum frequency value

# In[40]:


train_data["model"].unique()[0]


# Golf occuring max times , lets fill it with golf

# In[41]:


train_data["model"].fillna("golf",inplace =True)


# In[42]:


train_data.isnull().sum()


# Hurrah all nans have been removed

# In[43]:


train_data.head()


# ## Lets look at the postal code 

# In[45]:


train_data["postalCode"].head()


# Postalcode is not much giving much importance so lets remove this column

# In[46]:


del train_data["postalCode"]


# ## It's time to convert string into integers

# In[47]:


from sklearn.preprocessing import LabelEncoder


# In[74]:


data = train_data.copy()


# In[75]:


data["vehicleType"] =LabelEncoder().fit_transform(data["vehicleType"])
data["fuelType"] =LabelEncoder().fit_transform(data["fuelType"])
data["gearbox"] =LabelEncoder().fit_transform(data["gearbox"])
data["notRepairedDamage"] =LabelEncoder().fit_transform(data["notRepairedDamage"])
data["brand"] =LabelEncoder().fit_transform(data["brand"])
data["model"] =LabelEncoder().fit_transform(data["model"])
data["abtest"] =LabelEncoder().fit_transform(data["abtest"])


# ## Outliers Removal

# In[76]:


# analysis year o registration
data["yearOfRegistration"].describe()


# We can see yearOfRegistration goes from 1000 to 9999  but 50% of the values are nearby 1999

# In[77]:


data[data.yearOfRegistration > 2017].shape


# In[78]:


data[data.yearOfRegistration < 1950].shape


# So only 289 registration are there before 1950 and 4000 registration after 2017 , so these are outliers and should be removed for better results

# In[79]:


data = data[(data.yearOfRegistration < 2017)  & (data.yearOfRegistration > 1950)]


# In[82]:


# now lets look at the price
data["price"].describe()


# price starts from 0 and goes upto max 2.147484e+09 , but the price 0 is unrealistic

# In[86]:


data[data.price < 100].shape


# In[88]:


data[data.price > 200000].shape


# only 158 values are there above 2 lakh and 12000 enries below 100, lets remove them

# In[89]:


data = data[(data.price > 100) & (data.price < 200000) ]


# In[90]:


# lets seperate the output and input
y  = data["price"]
x =  data.drop("price",axis=1)


# ## Lets split the data into training and testing data

# In[91]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)


# In[92]:


# classifier
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr.score(x_test, y_test)


# ## Great , we got accuracy of 81% with randomforestregressor but we can get more by tweaking the paramets or using different classifiers

# ## The main focus of this notebook is on how to clean the data and outliers which can make our model inaccurate

# ## We can use feature scaling and can use different classifiers

# ## Thank you for looking the notebook , if you find this notebook useful then please upvote ..
