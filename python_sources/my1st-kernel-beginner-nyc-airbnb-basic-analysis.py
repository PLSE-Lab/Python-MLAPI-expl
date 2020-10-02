#!/usr/bin/env python
# coding: utf-8

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


# Have performed few of the basic analysis on the Airbnb dataset
# My main intention to do this is to help the beginners understand the problem and do some basic analysis
# So for the better understanding have put the code simple & straight & tried my best to explain each line by putting a comment next to it.
# 
# 
# 
# PERFORMED THE ANALYSIS **ONLY IN BEGINNERS PERSPECTIVE**
# 
# 
# 
# 
# 
# Lets dive into it........... :)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# as considering that you had a prior look at the give dataset description & content leme start it directly.

# **Lets import the necessary libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#now lets read/load the dataset into an variable using pandas
data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


#this function is used to display the first 5 rows from the data
data.head()


# **Let's do some basic data analysis**

# In[ ]:


#used to display the length of the whole dataset
len(data)


# In[ ]:


#displays the shape of our data set
data.shape


# In[ ]:


#this function is used to get the basic information about the dataset  
data.info()


# In[ ]:


#this gives the overall discription about the dataset such as mean,std,count....
data.describe()


# In[ ]:


#this function is used to display the correlation between the variables in the dataset
data.corr()


# # **Now lets do some EDA**

# In[ ]:


#lets check the datatypes present in the dataset
data.dtypes


# In[ ]:


#it helps us to know if we have any missing values in our dataset
data.isnull().sum()


# Here we can see that (name, host_name, last_review, reviews_per_month) this columns have the missing values

# lets drop few columns which are not that important/has more number of missing values.
# But removing columns which has the most missing values is not recommended always but, mostly it depends upon the kind of dataset given & to which
# domain it belongs too. In Some cases we see few columns with 50% of the data missing but if that column is highly needed & acts as an important feature for the given dataset **(based on the domain)** Then we definitely need to perform various methods to fill those missing values 

# So from the given data set i would like to remove ("id" "host_name" "last_review" "reviews_per_month") as i see id & host_name doesn't add up much value & "last_review" "reviews_per_month" have the more number of missing values so i choose to drop them 

# In[ ]:


#lets use drop function to drop the columns from the dataset
data.drop(["id","host_name","last_review","reviews_per_month"],axis = 1, inplace=True)


# In[ ]:


#now let's again check the shape of our data to ensure about the number of columns present after performing the drop function
data.shape


# In[ ]:


#shows the name of the columns present in the dataset
data.columns


# Now as we got the relevant columns needed, now lets explore some unique values present in the dataset for some specific columns 

# In[ ]:


#this unique() function helps us to find out the unique values present in the dataset
data.neighbourhood_group.unique()


# Here we can see that
# from the column **neighbourhood_group** we have few unique values present such as **'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx'** so this was possible to know with the use of above function

# In[ ]:


data.neighbourhood.unique()


# In[ ]:


data.room_type.unique()


# Now as we know the kind of unique values present in few columns, now lets know the count of those unique values, just to have an idea which might be usefull in some cases

# In[ ]:


#we do this with the help of value_counts() function
data.neighbourhood_group.value_counts()


# In[ ]:


data.neighbourhood_group.value_counts()


# In[ ]:


data.neighbourhood.value_counts().head()


# In[ ]:


data.room_type.value_counts()


# # **Now ending my EDA part here lets do some visualisation**
# for the dataset to get some insights out of it in a visual formate which inturn would be more easy to understand 

# In[ ]:


sns.countplot(x= 'neighbourhood_group',data = data)
plt.title("Popular neighbourhood_group ")
plt.show()


# From the above fig we can easily get to know that the most popular neighbourhood_groups are Manhattan & Brooklyn

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='room_type',data = data)
plt.title("Most occupied room_type")
plt.show()


# Here we can see that most occupied room_types are Entire home/apt & Private room 
# and comparitively shared rooms to be very low

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = data)
plt.title("Room types occupied by the neighbourhood_group")
plt.show()


# From the above fig we can get to know that in the category of **Private room occupancy -> neighbourhood_group : "Brooklyn"** seems to have the highest occupancy count and where as from the category of **Entire home/apt room occupancy -> neighbourhood_group :"Manhattan"** seems to have the highest occupancy count and in shared rooms its **Manhattan** again has an high count compared with Brooklyn.
#  
# 

# In[ ]:


ng = data[data.price <500]
plt.figure(figsize=(12,7))
sns.boxplot(y="price",x ='neighbourhood_group' ,data = ng)
plt.title("neighbourhood_group price distribution < 500")
plt.show()


# From the above box plot we can see that **Manhattan** has the highest pricing and it shows that it has an avg pricing range at 150 from the above observation & next comes **Brooklyn** with an avg price ranging at around 90 & then comes **Queens & Staten island** with the similar pricing at an avg 80 per night and in comparision with neighbourhood_group **Bronx** has the low pricing at an avg of 70

# # Now lets see the distribution of neighbourhood_group on given longitude,latitude

# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude',y='latitude',hue = "neighbourhood_group",data = data,palette = 'hls')
plt.show()


# Now lets see based upon the availability_365 feature.

# In[ ]:


plt.figure(figsize=(9,7))
sns.scatterplot(x='longitude',y='latitude',hue = 'availability_365',data = data, palette='coolwarm')
plt.show()


# This was not a complete analysis on the given data but yeah have tried to cover few basic steps & methods & tried to get few insights from it, HOPE it gives you some idea & give it an upvote if you find it easy & helpfull ....:) happy learning !

# In[ ]:




