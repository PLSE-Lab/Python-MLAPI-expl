#!/usr/bin/env python
# coding: utf-8

# #Inital few steps is for interacting with Kaggle dataset from Google colab
# #if not create a folder
# !mkdir .kaggle
# !ls -a

# #Add ur profile key and username to connect with Colab
# import json
# token = {"username":"<UserName>","key":"<KEY>"}
# with open('/content/.kaggle/kaggle.json', 'w') as file:
#     json.dump(token, file)

# #Additional settings
# !chmod 600 /content/.kaggle/kaggle.json

# !cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json

# !kaggle datasets download -d uciml/autompg-dataset

# #Unzip the zip file
# !unzip \autompg-dataset.zip
# 

# In[ ]:


import pandas as pd


# In[ ]:


auto = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")

auto.head(5)


# In[ ]:


#Check for datatypes and null values
auto.info()
#Observation - horsepower datatype is object need to dive deeper


# In[ ]:


import numpy as np
auto.horsepower.unique()
#Here hp has ? in the fields, we need to replace them


# In[ ]:


#Lets conver the unwanted symbol to nan
auto = auto.replace('?',np.nan)
auto.isna().sum()
#6 hp has missing values
#Need to conver hp to integer


# In[ ]:


auto['horsepower'] = pd.to_numeric(auto['horsepower'])
#Lets fill the na's with mean value
auto['horsepower'] = auto['horsepower'].fillna(auto['horsepower'].mean())


# In[ ]:


#Lets check the model year, Here for predicting milage model age will be needed than model year.
#Lets create a new feature
auto['model year'].unique()


# In[ ]:


import datetime
auto['model year'] = auto['model year']+1900 # To conver this to YYYY


# In[ ]:


auto.head(1)


# In[ ]:


auto['Age'] = (datetime.datetime.now().year)-auto['model year']
auto.head(1)


# In[ ]:


#Lets drop model 
auto.drop(['model year'],axis=1,inplace=True)


# In[ ]:


auto['car name'].unique()
#Car name also wont determine the milage we can drop it as well


# In[ ]:


auto.drop(['car name'],axis=1, inplace=True)


# In[ ]:


#At this point datacleaning is completed lets visualize the data now with MATPLOTLIB


# In[ ]:


#Datanow looks clean, lets perform some visualization
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(12,5))

plt.bar(auto['Age'],auto['mpg'])

plt.xlabel("Age")
plt.ylabel("Miles per galon")
plt.show()
# As age increase the MPG decreases


# In[ ]:


plt.scatter(auto['acceleration'],auto['mpg'])
plt.xlabel("acceleration")
plt.ylabel("Miles per galon")
#More the acceleration more the MPG


# In[ ]:


plt.scatter(auto['weight'],auto['mpg'])
plt.xlabel("weight")
plt.ylabel("Miles per galon")
# As weight increases milage decreases


# In[ ]:



auto.plot.scatter(x='weight',y='mpg')
plt.xlabel("weight")
plt.ylabel("Miles per galon")


# In[ ]:


plt.bar(auto['cylinders'],auto['mpg'])
plt.xlabel("cylinders")
plt.ylabel("Miles per galon")
#Not relation, can be droped


# In[ ]:


auto.drop(['cylinders','origin'],inplace=True, axis=1)


# In[ ]:


auto.head()


# In[ ]:


#Lets check the corelation
autocorr = auto.corr()
autocorr


# In[ ]:


# Lets plot this with heat map
import seaborn as sns
sns.heatmap(autocorr,annot=True)


# In[ ]:


#Lets apply the ML model
auto.shape


# In[ ]:


X = auto.drop('mpg',axis=1)
Y = auto['mpg']


# In[ ]:


#Lets apply simple linear regression to predict the age.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


linear_model = LinearRegression(normalize=True).fit(x_train,y_train)


# In[ ]:


print("Trainign score: ",linear_model.score(x_train,y_train))


# In[ ]:


y_pred = linear_model.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


print("test score : ", r2_score(y_test, y_pred))

