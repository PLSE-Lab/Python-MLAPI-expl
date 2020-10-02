#!/usr/bin/env python
# coding: utf-8

# **Hello Friends, I would like to share this notebook with kaggle users.**

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Loading Data

df = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv")
df.head()


# **CLEANING THE DATA**

# 1. Dropping "Unnamed: 0" column, since it's not useful
# 2. Dropping "Floor" column, since it has 26% of missing values and it's not usefull even after imputation

# In[ ]:


df = df.drop(["Unnamed: 0","floor"], axis = 1)
df.head()


# In[ ]:


#Checking for missing values

df.isnull().sum()


# Removing the dollor symbol and making the variables as numeric

# In[ ]:


def remove_dollor(x):
    a =  x[2:] #removes first two chr
    result = ""
    for i in a:
        if i.isdigit() is True:
            result = result + i
    return result #returns only digits (excludes special character)


# In[ ]:


df["hoa"] = pd.to_numeric(df["hoa"].apply(remove_dollor), errors= "ignore")
df["rent amount"] = pd.to_numeric(df["rent amount"].apply(remove_dollor), errors= "ignore")
df["property tax"] = pd.to_numeric(df["property tax"].apply(remove_dollor), errors= "ignore")
df["fire insurance"] = pd.to_numeric(df["fire insurance"].apply(remove_dollor), errors= "ignore")
df["total"] = pd.to_numeric(df["total"].apply(remove_dollor), errors= "ignore")


# In[ ]:


df.dtypes


# In[ ]:


df.head()


# **Checking Outliers**

# In[ ]:


plt.figure(figsize = (5,5))
sns.boxplot(df["total"])


# In[ ]:


#Since we found many outlliers, let's try to remove.

q1 = df["total"].quantile(0.25)
q3 = df["total"].quantile(0.75)

IQR = q3 - q1
IF = q1 - (1.5 * IQR)
OF = q3 + (1.5 * IQR)


# In[ ]:


data = df[~((df["total"] < IF) | (df["total"] > OF))]
data.shape


# In[ ]:


data.head()


# In[ ]:


print("Before Outlier Removal")
print("No. of rows : ", df.shape[0])
print("No. of columns : ", df.shape[1])
print("=======================")
print("After Outlier Removal")
print("No. of rows : ", data.shape[0])
print("No. of columns : ", data.shape[1])


# Check how Outliers are influencing the dataset

# In[ ]:


#Before Removing Outliers
plt.figure(figsize = (7,7))
sns.set(style = "whitegrid")
f = sns.barplot(x = "rooms", y = "total", data = df)
f.set_title("Before Removing Outliers")
f.set_xlabel("No. of Rooms")
f.set_ylabel("Total Cost")


# In[ ]:


#After Removing Outliers
plt.figure(figsize = (7,7))
sns.set(style = "whitegrid")
f = sns.barplot(x = "rooms", y = "total", data = data)
f.set_title("After Removing Outliers")
f.set_xlabel("No. of Rooms")
f.set_ylabel("Total Cost")


# Let's take more information from the dataset

# In[ ]:


df.columns


# In[ ]:


columns = ["city","rooms","bathroom","parking spaces", "animal", "furniture"]
plt.figure(figsize = (30,30))
for i,var in enumerate(columns,1):
    plt.subplot(2,4,i)
    f = sns.barplot(x = data[var], y = data["total"])
    f.set_xlabel(var.upper())
    f.set_ylabel("Total Cost")


# **INSIGHTS:**
# 1. Cost is more for houses present in *city*
# 2. Cost is more for houses which is *furnished* and *accepts pets*
# 3. Increasing in room, parkking spaces and bathrooms house cost is also increases

# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr(), annot=True)


# **Checking Normality**

# In[ ]:


plt.figure(figsize = (7,7))
sns.set(style = "whitegrid")
f = sns.distplot(data["total"])


# *It is right skewed, hence there will be no normality in the response variable*

# **ENCODING**

# In[ ]:


data["animal"].value_counts()


# In[ ]:


animal_dict = {"acept": 1,"not acept":0}
data["animal_en"] = data["animal"].map(animal_dict)
data.head()


# In[ ]:


data["furniture"].value_counts()


# In[ ]:


furniture_dict = {"furnished":1, "not furnished":0}
data["furniture_en"] = data["furniture"].map(furniture_dict)


# In[ ]:


data = data.drop(["animal","furniture"], axis = 1)


# In[ ]:


data.head()


# Now, **standardizing** the data

# In[ ]:


data["hoa"] = df["hoa"].fillna(data["hoa"].median())
data["property tax"] = data["property tax"].fillna(data["property tax"].median())


# In[ ]:


x = data.drop("total", axis = 1)
y = data["total"]


# In[ ]:


#Spliting train and test

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# Standardizing after splitting the data is always the good practice

# In[ ]:


#Standardization train data
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(x_train)
std_data = std.transform(x_train)


# **Box-Cox Transformation**: It is the way to transform non-normal dependent variables into a normal shape.

# In[ ]:


#Box-Cox transformation for train response variable
from scipy import stats
bx, lam = stats.boxcox(y_train)
y_total = bx


# **RANDOM FOREST REGRESSOR**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
#Training the model
rf = RandomForestRegressor(n_estimators = 1500)
rf.fit(std_data, y_total)


# In[ ]:


#Standardization test data
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(x_test)
std_test = std.transform(x_test)


# In[ ]:


#Prediction
pred = rf.predict(std_test)


# In[ ]:


#Box-Cox transformation for Test Response variable
bx, lam = stats.boxcox(y_test)
y_total_test = bx


# In[ ]:


def rmse_test(ytest, pred,xtest):
    err = ytest - pred
    mse = sum(err**2)/(xtest.shape[0]-xtest.shape[1]-1)
    rmse = np.sqrt(mse)
    print("RMSE OF TEST DATA IS : ", rmse)


# In[ ]:


rmse_test(y_total_test,pred,x_test)


# **Thanks** for reading my notebook. I'm new to Data Science, your one upvote will be my motivation :)
