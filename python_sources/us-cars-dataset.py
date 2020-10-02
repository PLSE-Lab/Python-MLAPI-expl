#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import all the necessary libraries
import numpy as np # numerical python
import pandas as pd # data structures and tool
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Import dataset to be used
df = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
df.head()


# Now, We will start with our Data Wrangling Process and Identify places for Data Cleaning
# 

# In[ ]:


df.dtypes
#All the datatypes are perfectly correct


# Task- To identify all the factors that affect Price of a Car.

# In[ ]:


#Lets delete all the columns which will be of no use to us.Eg- Vin, lot, unnmaed:0
df= df.drop(["vin","lot","Unnamed: 0"], axis= 1)


# In[ ]:


#Statistical summary to know our data
df.describe()


# Average Price of a car= $18767 with a very high standard deviation.
# Years are from 1973 to 2020
# There are 2499 entries in the dataset
# Avg. mileage= 5229.9 units with very high standard deviation

# In[ ]:


#Now let us start by analysing the behavior of Price
sns.distplot(df["price"])


# In[ ]:


#Now, Lets look our dependent variable nature with a few independent variables
y= df["price"]
x= df["year"]
plt.scatter(x,y)
plt.show()


# No as such relationship between year and price

# In[ ]:


#price vs brand - Which are the most expensive Cars in US?
df_1= df.groupby(["brand"]).price.mean()
df_2= df_1.sort_values(ascending= True)
df_2.plot(kind= "barh")


# In[ ]:


#Now lets explore which car is the most popular car in US?
df_3= df.groupby("brand").brand.count()
df_4= df_3.sort_values(ascending= False)
df_4


# Clearly, Ford followed by dodge are most popular cars in US.

# In[ ]:


#Relationship with Categorical Variables
#Color of Ford vs Price


# In[ ]:




