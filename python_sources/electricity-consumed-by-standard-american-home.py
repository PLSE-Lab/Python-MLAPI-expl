#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
print("All packages are installed successfully!")


# In[ ]:


print("Top 5 values of the dataset :")
data = pd.read_csv('../input/electricity-consumption/electricity_consumption.csv')
data.head()


# In[ ]:


print("Bottom 5 values of the dataset :")
data.tail()


# In[ ]:


print("The number of entries in this dataframe :",len(data))


# In[ ]:


print("Is there any null value in the dataframe ?",data.isnull().values.any())


# In[ ]:


print("Display a detailed list of null values in dataframe :")
print(data.isnull().sum())


# In[ ]:


print("Yearly records of 2016")
data_16 = data.iloc[:12,:]
data_16


# In[ ]:


data_16.plot(kind='bar',x='Usage_charge',y='Billed_amount')
plt.title('Billed amount per unit Usage charge')


# In[ ]:


plt.plot(data_16['On_peak'].unique())
plt.xlabel('no. of months')
plt.plot(data_16['Off_peak'].unique())
plt.ylabel('range of values of peaks')
plt.title('Comparative analysis of On_peak and Off_Peak data\n ')
plt.show()


# **Off_peak graph is M shaped which suggests that the maximum electricity consumption takes place during 6th month, reduces during 7th month and again rises during 8th month in 2016**

# In[ ]:


print("Yearly records of 2017")
data_17 = data.iloc[12:24,:]
data_17


# In[ ]:


data_17.plot(kind='bar',x='Usage_charge',y='Billed_amount')
plt.title('Billed amount per unit Usage charge')


# In[ ]:


plt.plot(data_17['On_peak'].unique())
plt.xlabel('no. of months')
plt.plot(data_17['Off_peak'].unique())
plt.ylabel('range of values of peaks')
plt.title('Comparative analysis of On_peak and Off_Peak data\n ')
plt.show()


# **Off_peak graph is M shaped which suggests that the maximum electricity consumption takes place during 6th month, reduces during 7th month and again rises during 8th month in 2017 also, but after the 10th month there has been a shear rise in consumption**

# In[ ]:


print("Yearly records of 2018")
data_18 = data.iloc[24:36,:]
data_18


# In[ ]:


data_18.plot(kind='bar',x='Usage_charge',y='Billed_amount')
plt.title('Billed amount per unit Usage charge')


# In[ ]:


plt.plot(data_18['On_peak'].unique())
plt.xlabel('no. of months')
plt.plot(data_18['Off_peak'].unique())
plt.ylabel('range of values of peaks')
plt.title('Comparative analysis of On_peak and Off_Peak data\n ')
plt.show()


# **The graph starts from a high value and gradually decreases step-wise till the 4th month and again rises step-wise to the apex in the 9th month. After that the consumption again falls in 2018**

# In[ ]:


print("Yearly records of 2019")
data_19 = data.iloc[36:48,:]
data_19


# In[ ]:


data_19.plot(kind='bar',x='Usage_charge',y='Billed_amount')
plt.title('Billed amount per unit Usage charge')


# In[ ]:


plt.plot(data_19['On_peak'].unique())
plt.xlabel('no. of months')
plt.plot(data_19['Off_peak'].unique())
plt.ylabel('range of values of peaks')
plt.title('Comparative analysis of On_peak and Off_Peak data\n ')
plt.show()


# **This graph look like a W-A. It starts with a standard value from the 1st month, reduces during 2nd to 5th and again gradually rises to a peak in 8th month. After this the consumption again falls to 10th month with a slight hike in 11th month in 2019**

# In[ ]:


print("Yearly records of 2020 till now ")
data_20 = data.iloc[48:52,:]
data_20


# In[ ]:


data_20.plot(kind='bar',x='Usage_charge',y='Billed_amount')
plt.title('Billed amount per unit Usage charge')


# In[ ]:


plt.plot(data_20['On_peak'].unique())
plt.xlabel('no. of months')
plt.plot(data_20['Off_peak'].unique())
plt.ylabel('range of values of peaks')
plt.title('Comparative analysis of On_peak and Off_Peak data\n ')
plt.show()


# **Till now the Off_peak online declines, but most probably it will again rise in the 6th to 8th month and fall by the 10th**

# In[ ]:


plt.plot(data_16['On_peak'].unique())
plt.plot(data_17['On_peak'].unique())
plt.plot(data_18['On_peak'].unique())
plt.plot(data_19['On_peak'].unique())
plt.plot(data_20['On_peak'].unique())
plt.title('Comparative analysis of On_peak values for each year ')
plt.show()


# **It is noticable that all graphs show a general steep rise during the 7th to 8th month and a fall after that. So from this we can predict that year 2020 will also be the same.**

# 

# In[ ]:


data.corr()


# In[ ]:


plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(),annot=True,fmt="f").set_title("Corelation of attributes of Electricity Consumption Data")
plt.show()


# # So, we can notice from this heatmap that Off_peak values depend the most on Usage_charge and On_peak values depend the most on the Billed_amount

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data.iloc[:,1:3]
y = data.iloc[:,3]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# # This project is under construction.

# In[ ]:




