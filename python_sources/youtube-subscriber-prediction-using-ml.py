#!/usr/bin/env python
# coding: utf-8

# # YouTube Subscriber Prediction using ML

# In this kernel we will use data about top 5000 YouTube channels from socialblade to do some basic analysis.

# ### **Importing packages**

# In[ ]:


##Importing the packages
#Data processing packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualization packages
import matplotlib.pyplot as plt 
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')

#Machine Learning packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ### **Importing the data**

# In[ ]:


data = pd.read_csv('../input/data.csv')


# ### **Basic Analysis**

# In[ ]:


#Find the size of the data Rows x Columns
data.shape


# **COMMENTS:** The data consists of 5000 rows and 6 columns

# In[ ]:


#Display first 5 rows of Customer Data
data.head()


# **COMMENTS:** The data consists of Name of YouTube channel, ranking, grades and other statistics like no. of Video uploads, Subscribers and Video views

# In[ ]:


#Find the the information about the fields, field datatypes and Null values
data.info()


# In[ ]:


#Convert datatype to numerical (float64, int64)
data['Subscribers'] = pd.to_numeric(data['Subscribers'], errors='coerce')
data['Video Uploads'] = pd.to_numeric(data['Video Uploads'], errors='coerce')
data['Video views'] = pd.to_numeric(data['Video views'], errors='coerce')


# In[ ]:


#Check and confirm data type has become numerical for the fields Subscribers, Video Uploads and Video views
data.info()


# **COMMENTS:** From the above output we see that there are missing values in **Video uploads and Subscribers**

# In[ ]:


#Find Basic Statistics like count, mean, standard deviation, min, max etc.
data.describe()


# ### **YouTube Channel count by Grade**

# In[ ]:


#Find count of youtube channels by Grade
sns.countplot(data=data, x='Grade')
plt.title('Grade')
data.Grade.value_counts()


# **COMMENTS:** Grades are inversely proportional to Channel count. Higher the grade (A+, A++), lower the channel count in that category.

# ### **Extract top 20 YouTube channels**

# In[ ]:


#Data is already arranged in ascending order of Ranks.  If we select first 20 then we will get top 20 Ranks
data_top_20 = data.head(20)


# ### **Find the count of Top 20 channels by Grades**

# In[ ]:


sns.set(style='whitegrid')
sns.countplot(data=data_top_20, x='Grade')
plt.title('Grade')
data_top_20.Grade.value_counts()


# ### **Plotting the YouTube channels based on Subscribers**

# In[ ]:


plt.figure(figsize=(18,6))
p = sns.barplot(data=data_top_20.sort_values('Subscribers',ascending=False), x='Channel name', y='Subscribers')
plt.title('YouTube Channel Name     vs       Subscribers')
b = plt.setp(p.get_xticklabels(), rotation=45)
data_top_20.sort_values('Subscribers',ascending=False)


# ### **Plotting the YouTube channels based on Video uploads**

# In[ ]:


plt.figure(figsize=(18,6))
p = sns.barplot(data=data_top_20.sort_values('Video Uploads',ascending=False), x='Channel name', y='Video Uploads')
plt.title('YouTube Channel Name     vs       Video Uploads')
_ = plt.setp(p.get_xticklabels(), rotation=45)
data_top_20.sort_values('Video Uploads',ascending=False)


# ### **Plotting the YouTube channels based on Video views**

# In[ ]:


plt.figure(figsize=(18,6))
p = sns.barplot(data=data_top_20.sort_values('Video views',ascending=False), x='Channel name', y='Video views')
plt.title('YouTube Channel Name     vs       Video views')
b = plt.setp(p.get_xticklabels(), rotation=45)
data_top_20.sort_values('Video views',ascending=False)


# ### **Plotting the Subscribes vs Video views**

# In[ ]:


#Subscribers vs Video views
plt.figure(figsize=(18,6))
plt.subplot(121); sns.scatterplot(x="Subscribers", y="Video views", data=data_top_20)
plt.title('Subscribers   vs   Video views [Scatter Plot]')
plt.subplot(122); sns.regplot(x="Subscribers", y="Video views", data=data_top_20)
plt.title('Subscribers   vs   Video views [Regression Plot]')


# In[ ]:


#Subscribers vs Video Uploads
plt.figure(figsize=(18,6))
plt.subplot(121); sns.scatterplot(x="Subscribers", y="Video Uploads", data=data_top_20)
plt.title('Subscribers   vs   Video Uploads [Scatter Plot]')
plt.subplot(122); sns.regplot(x="Subscribers", y="Video Uploads", data=data_top_20)
plt.title('Subscribers   vs   Video Uploads [Regression Plot]')


# ### **Separating the Feature and Target Matrices**

# In[ ]:


X = data[['Video Uploads', 'Video views']]
y = data[['Subscribers']]


# ### **Fill the missing values in _Video uploads and Subscribers_**

# In[ ]:


#SimpleImputer is Imputation transformer for completing missing values.
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
y = my_imputer.fit_transform(y)


# ### **Split the data into Training set and Testing set**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### **Train the _Linear Regression_ Model**

# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)


# ### **Test (make predictions) the _Linear Regression_ Model**

# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:




