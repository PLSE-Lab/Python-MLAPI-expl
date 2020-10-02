#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.chdir('../input/')
os.getcwd()
os.listdir()


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


my_data=pd.read_csv("train(1).csv",index_col=['Id'])


# In[ ]:


my_data.head(20)


# In[ ]:


my_data.describe()


# In[ ]:


my_data.info()


# In[ ]:


my_data.shape


# In[ ]:


my_data.columns


# # **visualization**

# In[ ]:


#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.
my_data['Wild_area'] = (my_data.iloc[:, 10:14] == 1).idxmax(1)
my_data['Soil_type'] = (my_data.iloc[:, 15:55] == 1).idxmax(1)


# In[ ]:


sns.scatterplot(x=my_data['Elevation'],y=my_data['Horizontal_Distance_To_Roadways'],hue=my_data['Cover_Type'],palette='rainbow');


# In[ ]:


sns.scatterplot(x=my_data['Slope'],y=my_data['Hillshade_Noon'])


# In[ ]:


sns.scatterplot(x=my_data['Elevation'],y=my_data['Horizontal_Distance_To_Roadways']);


# In[ ]:


sns.scatterplot(x=my_data['Aspect'],y=my_data['Hillshade_9am']);


# # **Categorical Data Exploration**

# 

# In[ ]:


#Count of the entries from different wilderness areas.
plt.figure(figsize=(10,10))
sns.countplot(my_data['Wild_area']);


# Entries from Comanche Peak Wilderness Area occur the most and entries from the Neota Wilderness Area, the least

# In[ ]:


plt.figure(figsize=(50,8))
sns.countplot(my_data['Soil_type']);


# soil_type10 has more count value and soil type 25 is very rare.

# In[ ]:


plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)

plt.subplot(5,2,1)
a1=sns.boxplot(y=my_data['Elevation'],x=my_data['Cover_Type']);

plt.subplot(5,2,2)
sns.boxplot(y=my_data['Aspect'],x=my_data['Cover_Type']);

plt.subplot(5,2,3)
#Boxplot between Slope and Cover type
sns.boxplot(y=my_data['Slope'],x=my_data['Cover_Type'],palette='rainbow');


plt.subplot(5,2,4)
#Boxplot between Horizontal_Distance_To_Hydrology and Cover type
sns.boxplot(y=my_data['Horizontal_Distance_To_Hydrology'],x=my_data['Cover_Type'],palette='rainbow');

plt.subplot(5,2,5)
#Boxplot between Vertical_Distance_To_Hydrology and Cover type
sns.boxplot(y=my_data['Vertical_Distance_To_Hydrology'],x=my_data['Cover_Type'],palette='rainbow');


plt.subplot(5,2,6)
#Boxplot between Horizontal_Distance_To_Roadways and Cover type
sns.boxplot(y=my_data['Horizontal_Distance_To_Roadways'],x=my_data['Cover_Type'],palette='rainbow');

plt.subplot(5,2,7)
#Boxplot between Hillshade_9am and Cover type
sns.boxplot(y=my_data['Hillshade_9am'],x=my_data['Cover_Type'],palette='rainbow');

plt.subplot(5,2,8)
#Boxplot between Hillshade_Noon and Cover type
sns.boxplot(y=my_data['Hillshade_Noon'],x=my_data['Cover_Type'],palette='rainbow');

plt.subplot(5,2,9)
#Boxplot between Hillshade_3pm and Cover type
sns.boxplot(y=my_data['Hillshade_3pm'],x=my_data['Cover_Type'],palette='rainbow');

plt.subplot(5,2,10)
#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type
sns.boxplot(y=my_data['Horizontal_Distance_To_Fire_Points'],x=my_data['Cover_Type'],palette='rainbow');


# 

# 

# 

# In[ ]:


plt.figure(figsize=(16,16))
corrMetrix=my_data.iloc[:,:10].corr()
sns.heatmap(corrMetrix,vmin=-1,cmap='coolwarm',annot=True,square=True)
plt.show()


# A correlation matrix of the 10 numerical variables is created and plotted:
# There are six pairwise correlations have value>absolute 0.5
# 
# 
# 
#     HS.noon, HS.3pm (0.61)
#     HD.9am, HS.3pm (-0.78)
#     HD.Hyrdro, VD.Hyrdo (0.65)
#     Slope, HS.noon (-0.61)
#     Aspect, HS.9am (-0.59)
#     Aspect, HS.3pm (0.64)
#     Elevation, HD.Road (0.58)
# 

# In[ ]:


my_data_deg=my_data[['Elevation','Aspect','Slope','Cover_Type']]


# In[ ]:


sns.pairplot(my_data_deg,hue='Cover_Type')


# # **Swarmplot **

# In[ ]:


plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)


plt.subplot(2,2,1)
a1=sns.swarmplot(data=my_data,x='Wild_area',y='Slope',hue='Cover_Type')

plt.subplot(2,2,3)
a2=sns.swarmplot(data=my_data,x='Wild_area',y='Elevation',hue='Cover_Type')

plt.subplot(2,2,2)
a3=sns.swarmplot(data=my_data,x='Cover_Type',y='Slope')


# # **Numerical Data Exploration**

# In[ ]:


plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)

plt.subplot(3,2,1)
a1=sns.kdeplot(data=my_data['Elevation'],shade=True)
#a1.set_xticklabels(a1.get_xticklabels(),rotation=15)

plt.subplot(3,2,3)
a2=sns.kdeplot(data=my_data['Slope'],shade=True)
#a2.set_xticklabels(a2.get_xticklabels(),rotation=15)

plt.subplot(3,2,5)
a3=sns.kdeplot(data=my_data['Aspect'],shade=True);
#a3.set_xticklabels(a3.get_xticklabels(),rotation=15)


plt.subplot(3,2,2)
a4=sns.kdeplot(data=my_data['Horizontal_Distance_To_Fire_Points'],shade=True)
#a4.set_xticklabels(a4.get_xticklabels(),rotation=15)

plt.subplot(3,2,4)
a5=sns.kdeplot(data=my_data['Horizontal_Distance_To_Roadways'],shade=True);
#a5.set_xticklabels(a5.get_xticklabels(),rotation=15)

plt.subplot(3,2,6)
a6=sns.kdeplot(data=my_data['Horizontal_Distance_To_Hydrology'],shade=True)
#a6.set_xticklabels(a6.get_xticklabels(),rotation=15)


# Elevation has a relative normal distribution
# 
# Aspect contains two normal distribution 
# 
# Slope, HD.Hyrdo, HD.Road have similar distribution

# In[ ]:


#Distribution of elevation values in the data.
sns.distplot(my_data['Elevation'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)


# The elevation values range between 2000m to 4000m. Thus the data is collected from relatively high altitude areas, as indicated by the vegetation type observed in the data. The distribution of elevation values follows a trimodal fashion, peaking in the middle of the intervals 2000m-2500m, 2500m-3000m,3000m-3500m, and tapering at both ends.

# In[ ]:


#Distribution of frequency of various soil types in the data.
my_data['Soil_type'].value_counts().plot(kind='barh',figsize=(10,10));
plt.xlabel('Frequency',fontsize=10)


# Of the fourty soil types mentioned,2, namely soil-type 8 and soil-type 25 are not present in this data.
# 
# Soil type 28 is present in the smallest amount.
# 
# Soil-type 10 is present in the maximum samples.
# 
# This indicates the wide differences in the representaion of various soil types in the region of interest.
# 

# 
