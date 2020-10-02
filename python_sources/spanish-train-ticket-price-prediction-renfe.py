#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor


# # Loading the dataset

# In[ ]:


data = pd.read_csv('../input/renfe.csv', index_col=0)
data.head()


# In[ ]:


data.info()


# # Checking for Null values in the dataset

# In[ ]:


data.isnull().sum()


# ## Filling the Null values in the price column by taking the mean price of the ticket

# In[ ]:


data['price'].fillna(data['price'].mean(),inplace=True)


# ## Dropping the rows containing Null values in the attributes train_class and fare

# In[ ]:


data.dropna(inplace=True)


# ## Dropping irrelevant attributes

# In[ ]:


data.drop('insert_date',axis=1,inplace=True)


# In[ ]:


data.isnull().sum()


# # Univariate Analysis

# ## 1. Number of people boarding from different stations

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(data['origin'])
plt.show()


# From the above graph we can visualize that maximum number of the people have **"Madrid"** as the station of origin.

# ## Number of people having the following stations as destination

# In[ ]:


fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(data['destination'])
plt.show()


# * From the above graph we can visualize that also maximum number of people are coming to **"Madrid"** as the most of the people have their destination station as **Madrid**.

# ## Different types of train that runs in Spain 

# In[ ]:


fig,ax = plt.subplots(figsize=(15,6))
ax = sns.countplot(data['train_type'])
plt.show()


# We can see that **"AVE"** are runs maximum in number as compared to other train types.

# ## Number of train of different class

# In[ ]:


fig,ax = plt.subplots(figsize=(15,6))
ax = sns.countplot(data['train_class'])
plt.show()


# **"Turista"** is the train_class in which people travel in general. 

# ## Number of tickets bought from each category 

# In[ ]:


fig,ax = plt.subplots(figsize=(15,6))
ax = sns.countplot(data['fare'])
plt.show()


# ## Distribution of the ticket prices

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.distplot(data['price'],rug=True)
plt.show()


# # 2. Bivariate Analysis

# ## train_class vs price

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='train_class',y='price',data=data)
plt.show()


# **"Cama G. Clase"** is the train class with the highest ticket price, and the tickets of this class are bought by least number of people.

# ## train_type vs price

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='train_type',y='price',data=data)
plt.show()


# The average price of the tickets of train_type **AVE and AVE-TGV** are comparatilvely higher as compared to other train types.

# # Feature Engineering

# ## Finding the travel time between the place of origin and destination

# In[ ]:


data = data.reset_index()


# In[ ]:


datetimeFormat = '%Y-%m-%d %H:%M:%S'
def fun(a,b):
    diff = datetime.datetime.strptime(b, datetimeFormat)- datetime.datetime.strptime(a, datetimeFormat)
    return(diff.seconds/3600.0)
    


# In[ ]:


data['travel_time_in_hrs'] = data.apply(lambda x:fun(x['start_date'],x['end_date']),axis=1) 


# ## Removing redundant features

# In[ ]:


data.drop(['start_date','end_date'],axis=1,inplace=True)
data.head()


# # 1. Travelling from MADRID to SEVILLA

# In[ ]:


df1 = data[(data['origin']=="MADRID") & (data['destination']=="SEVILLA")]
df1.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x="train_type",y="price",data=df1)
plt.show()


# - The fastest train between Madrid and SEVILLA is AVE and even the costliest one and it takes approximately 2-2.4 hrs.
# - The cheapest train is MD-LD , even the slowest one and it takes above 7 hours to reach the destination.

# # 2. Travelling from MADRID to BARCELONA

# In[ ]:


df1 = data[(data['origin']=="MADRID") & (data['destination']=="BARCELONA")]
df1.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x="train_type",y="price",data=df1)
plt.show()


# - The fastest trains on this route are AVE and AVE-TGV as they around 2-3 hours to reach the destination.
# - R.Express takes maximum time i.e. more than 8 hours and it is one of the cheapest one.

# # 3. Travelling from MADRID to VALENCIA

# In[ ]:


df1 = data[(data['origin']=="MADRID") & (data['destination']=="VALENCIA")]
df1.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x="train_type",y="price",data=df1)
plt.show()


# - AVE and ALVIA are two train_types which takes least amount of time to reach the destination, while MD-LD  takes maximum amount of time.
# - REGIONAL train have minimum ticket fare.
# - AVE is the best train to travel as it is the fastest train with less ticket pricing.

# # 4. Travelling from MADRID to PONFERRADA

# In[ ]:


df1 = data[(data['origin']=="MADRID") & (data['destination']=="PONFERRADA")]
df1.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="train_type",y="travel_time_in_hrs",data=df1)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x="train_type",y="price",data=df1)
plt.show()


# - It takes minimum 4 hours to travel from MADRID to PONFERRADA
# - AVE-MD , AVE-LD , ALVIA they take almost same time which is the minimum time and also their ticket prices are same
# - There is no point travelling via MD-LD as it takes maximum time and have the most expensive tickets.

# # Label Encoding

# In[ ]:


lab_en = LabelEncoder()
data.iloc[:,1] = lab_en.fit_transform(data.iloc[:,1])
data.iloc[:,2] = lab_en.fit_transform(data.iloc[:,2])
data.iloc[:,3] = lab_en.fit_transform(data.iloc[:,3])
data.iloc[:,5] = lab_en.fit_transform(data.iloc[:,5])
data.iloc[:,6] = lab_en.fit_transform(data.iloc[:,6])


# In[ ]:


data.head()


# # Splitting the data into training and test set

# In[ ]:


X = data.iloc[:,[1,2,3,5,6,7]].values
Y = data.iloc[:,4].values


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=5)


# # Applying Linear Regression

# In[ ]:


lr = LinearRegression()
lr.fit(X_train,Y_train)


# In[ ]:


lr.score(X_test,Y_test)


# # Applying Gradient Boosting

# In[ ]:


lg = LGBMRegressor(n_estimators=1000)
lg.fit(X_train,Y_train)


# In[ ]:


lg.score(X_test,Y_test)

