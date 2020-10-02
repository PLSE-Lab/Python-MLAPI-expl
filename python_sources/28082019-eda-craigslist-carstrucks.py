#!/usr/bin/env python
# coding: utf-8

# # USE CASE - EDA Used Cars Dataset

# ### Description
# 
# 1. url = Link to listing
# 2. city = Craigslist region in which this listing was posted
# 3. city_url = Link to region page
# 4. price = Price of vehicle
# 5. year = Year of vehicle
# 6. manufacturer = Manufacturer of vehicle
# 7. make = Make of vehicle
# 8. condition = Condition of vehicle
# 9. cylinders = Number of cylinders of vehicle
# 10. fuelFuel =  taken by vehicle
# 11. odometer = Miles vehicle has been driven
# 12. title_status = Title status of vehicle (e.g. clean - this vehicle has all legal documents. missing - these documents are missing)
# 13. transmission = Transmission of vehicle
# 14. VIN = Vehicle Identification Number
# 15. drive = Drive of vehicle
# 16. size = Size of vehicle
# 17. type = Vehicle type
# 18. paint_color = Color of vehicle
# 19. image_url = Link to image of vehicle
# 20. desc = Listing description provided by owner
# 21. lat = Latitude of vehicle (not precise but very close)
# 22. long = Longitude of vehicle (not precise but very close)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data Preparation

# ### Import Data

# In[ ]:


#import numpy and pandas
import numpy as np
import pandas as pd

#import visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

#import preprocessing
from sklearn import preprocessing


# In[ ]:


#import data craigslistVehiclesFull.csv
cvehiclesfull = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv')


# In[ ]:


#data information
cvehiclesfull.info()


# In[ ]:


#check length of columns and rows
cvehiclesfull.shape


# In[ ]:


#check statistical details
cvehiclesfull.describe()


# ### Cleansing

# In[ ]:


#detect missing value
cvehiclesfull.isnull().sum().sort_values(ascending=False) 


# Dataset has so many missing value in almost all variable, except lat, state_name, long, year, price, city, and url. We must fix it by handle missing value, because it is important to be handled as they could lead to wrong prediction or classification for any given model being used.

# In[ ]:


#change data type of year from float to string
cvehiclesfull['year'] = cvehiclesfull.year.astype(str)
cvehiclesfull.info()


# In[ ]:


#detect percentage of missing value which above 60%
checknull = round(cvehiclesfull.isnull().sum()/len(cvehiclesfull)*100,2).sort_values(ascending=False)
checknull


# Percentage missing value of variable size and vin is more than 60%. So that we should drop variable size and vin drop from the table to avoid the bias.

# In[ ]:


#drop column url, image url, size, and vin
#drop url and image_url because unique value
df = cvehiclesfull.drop(columns=['url','image_url', 'size', 'vin'])
df.head()


# In[ ]:


#separate numerical and categorical feature
category = ['city','manufacturer','make','cylinders','fuel','title_status','transmission','drive','type','paint_color','county_name','condition','state_code','state_name','year']
numerical = df.drop(category, axis=1)
categorical = df[category]
numerical.head()


# In[ ]:


categorical.head()


# In[ ]:


categorical.info()


# In[ ]:


numerical.info()


# In[ ]:


#fill value in numerical with mean
for num in numerical:
    mean = numerical[num].mean()
    numerical[num]=numerical[num].fillna(mean) 


# This simple imputation method is based on treating every variable individually, ignoring any interrelationships with other variables.This method is beneficial for simple linear models and NN. <br>
# **MEAN:** Suitable for continuous **data without outliers**

# In[ ]:


#detect numerical missing value
numerical.isnull().sum().sort_values(ascending=False) 


# In[ ]:


#fill value in categorical with mode
for cat in categorical:
    mode = categorical[cat].mode().values[0]
    categorical[cat]=df[cat].fillna(mode)


# In[ ]:


#detect categorical missing value
categorical.isnull().sum().sort_values(ascending=False) 


# In[ ]:


categorical.head()


# In[ ]:


numerical.head()


# ### Outlier Detection
# Anomaly detection (or **outlier detection**) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.

# In[ ]:


#concat table categorical and numerical to create new table without missing values
df2 = pd.concat([categorical,numerical],axis=1)
df2.head()


# In[ ]:


#outlier detection in numerical
fig=plt.figure(figsize=(13,12))
axes=330
#put data numerical
for num in numerical:
    axes += 1
    fig.add_subplot(axes)
    #set title of num
    sns.boxplot(data = numerical, x=num, color="y") 
plt.show()


# Based on boxplot visualization for numeric feature. We know that variable long, lat, and odomater has too many outliers, so that we can drop it from the table. Moreover we can see that price has so many outlier too, we can manipulate that variable

# In[ ]:


filterprice=df2[df2['price']<15000]
fig=plt.figure(figsize=(15,5))
fig.add_subplot(1,2,1)
sns.boxplot(numerical['price'])
plt.title('Price before dropped under 15000')
fig.add_subplot(1,2,2)
sns.boxplot(filterprice['price'], color="y")
plt.title('Price after dropped under 15000')


# In[ ]:


#drop column long lat and odometer because too many outliers
dfinal = df2.drop(columns=['long', 'lat', 'odometer'])
dfinal.head()


# ## Perform Visualization
# **Data visualization** refers to the techniques used to communicate data or information by encoding it as visual objects (e.g., points, lines or bars) contained in graphics. The goal is to communicate information clearly and efficiently to users. It is one of the steps in **data analysis** or **data science**.

# ### Business Question
# 1. Highest price based on type of car
# 2. Highest price based on manufacturer of car in 2010
# 3. Trend manufacturer base on type=sedan in 2010
# 2. How many production of cars type in 2010?
# 3. How many type of cars?
# 4. How many county_fips for car with 6 cylinder engine?
# 5. Manufacturer based on price

# In[ ]:


#filter by year 2010
data_year = dfinal[dfinal['year']=='2010.0']
top=data_year.sort_values('price',ascending=False).head(5)
toplabel=top[['manufacturer','price']]

plt.figure(figsize=(12,6))

x=range(5)
plt.bar(x,top['price']/6**9, color=['y', 'y', 'y', 'y', 'y'])
plt.xticks(x,top['manufacturer'])
plt.xlabel('Manufacturer of Cars')
plt.ylabel('Price')
plt.title('5 Most Highest Price of Manufacturer Cars in 2010')
plt.show()
toplabel.head()


# #### 1. Highest Price Based on Type Car

# In[ ]:


joins = dfinal[['manufacturer','type','price']]
join_group = joins.groupby('type').mean().head(5)
join_group


# In[ ]:


plt.figure(figsize=(12,6))

x=range(5)
plt.bar(x,join_group['price']/6**9, color="y")
plt.xticks(x,join_group.index)
plt.xlabel('Type of Cars')
plt.ylabel('Price')
plt.title('5 Most Highest Price Based on Type of Cars')
plt.show()


# ### Pearson Correlation with Heatmap

# In[ ]:


#create correlation with hitmap

#create correlation
corr = dfinal.corr(method = 'pearson')

#convert correlation to numpy array
mask = np.array(corr)

#to mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(20,5)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True, color="y")


# ### Scatter Plot 3D

# In[ ]:


data_type = dfinal[dfinal['type']=='sedan']

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_type['year'],data_type['county_fips'],data_type['price'], s=30, color="y")
plt.show()


# ### Distribution Plot
# This distribution plot for **numerical features** to know distribution of the each variables

# In[ ]:


fig=plt.figure(figsize=(15,10))
fig.add_subplot(2,2,1)
sns.distplot(filterprice['county_fips'], color="y", kde=False)
plt.title('Histogram of Federal Information Processing Standards code')

fig.add_subplot(2,2,2)
sns.distplot(filterprice['weather'], color="y", kde=False)
plt.title('Histogram of historical average temperature for location in October/November')

fig.add_subplot(2,2,3)
sns.boxplot(filterprice['price'], color="y")
plt.title('Histogram of vehicles price')

plt.show()


# In[ ]:




