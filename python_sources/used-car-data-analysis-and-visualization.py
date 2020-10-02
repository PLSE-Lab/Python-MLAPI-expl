#!/usr/bin/env python
# coding: utf-8

# # Introduction

# OLX Car Data contain nearly about 25000 rows and 9 columns

# In this Notebook I am going to analyse and visualize data from Pakistan Used Car data.I have choosen this data because there are no good work on this data and it is good chance to show my little skill on Data Analysis.I hope you will gain some insight from my work.

# # Data collection and Pre-processing

# In this phase I am going to load required libraries,Import data and use some function like head,tail,dtypes,describe the dataset.After successful exploration of data i try to make data clean by fixing data types issues,Dealing with null values and apply some technique to validate the data.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/OLX_Car_Data_CSV.csv', delimiter=',', encoding = "ISO-8859-1")
df.head()


# In[ ]:


print(df.columns)


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


df.describe(include = "all")


# In[ ]:


df.info


# In[ ]:


missing_data = df.isnull()
missing_data.head(5)


# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  


# Here we can see that except Price ,there are all columns have null values.So,I am going to Calculate avrage value and replace them with mean of their value.And also replacing some values with NaN values.

# In[ ]:


avg_KMsDriven = df["KMs Driven"].astype("float").mean(axis = 0)
print("Average of KMs Driven:", avg_KMsDriven)
df["KMs Driven"].replace(np.nan, avg_KMsDriven, inplace = True)


# In[ ]:


avg_Year = df["Year"].astype("float").mean(axis = 0)
print("Average of Year:", avg_Year)
df["Year"].replace(np.nan, avg_Year, inplace = True)


# In[ ]:


# simply drop whole row with NaN in "price" column
df.dropna(subset=["Brand","Condition", "Fuel","Model", "Registered City","Transaction Type"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


# In[ ]:


df.replace("", np.nan, inplace = True)
df.head(5)


# # Visualization And Data Analysis

# At this phase I am going to visualize the given data and try to gain some insights about the data set.

# In[ ]:


df['Fuel'].value_counts()


# In[ ]:


var = df.groupby('Fuel').Price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Fuel')
ax1.set_ylabel('Price')
ax1.set_title("Fuel Vs Price")
var.plot(kind='bar')


# In[ ]:


df['Condition'].value_counts()


# In[ ]:


var = df.groupby('Condition').Price.sum() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Condition Of Car')
ax1.set_ylabel('Increase In price')
ax1.set_title("Condition Vs Price")
var.plot(kind='bar')


# In[ ]:


df['Brand'].value_counts()


# In[ ]:


var = df.groupby('Brand').Price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Brand')
ax1.set_ylabel('Increase In price')
ax1.set_title("Brand Vs Price")
var.plot(kind='bar')


# In[ ]:


df['Model'].value_counts()


# In[ ]:


var = df.groupby('Model').Price.sum() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Model Of Car')
ax1.set_ylabel('Increase In price')
ax1.set_title("Model Vs Price")
var.plot(kind='line')


# In[ ]:


df['Year'].value_counts()


# In[ ]:


var = df.groupby('Year').Price.sum() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Condition Of Car')
ax1.set_ylabel('Increase In price')
ax1.set_title("Year Vs Price")
var.plot(kind='line')


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['KMs Driven'],df['Price']) 
plt.show()


# In[ ]:


df['Transaction Type'].value_counts()


# In[ ]:


var = df.groupby('Transaction Type').Price.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Transaction Type')
ax1.set_ylabel('Increase In price')
ax1.set_title("Transaction Type Vs Price")
var.plot(kind='bar')


# ## After visualizing and analysing the data we can say that :-

# -  Petrol cars are the most number of car for sale and have high price too
# -  Number of used car is more then new car for sale
# -  Suzuki, Toyota and Honda car Brand are in top 3 car for sell
# -   Less driven car has high price
# -  Cultus VXR,Alto and Corolla GLI car model are top 3 car model for sale
# -  Most are the car registered for sale in the year between 2000-2019
# -  Most of payment is done through cash                      
# -  Variables like Model,Year,KMs Driven and Fuel are good pridictor and positively correlated with price
# -  Variables like Transaction Type and Condition of car does not adding good corelation with price,hence negatively corilated.

# # Future work

# Now the data is preproccessed and clean we can apply different Machine learning algorithm to pridict the price of the car. 
