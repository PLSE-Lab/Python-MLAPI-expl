#!/usr/bin/env python
# coding: utf-8
# Data Understanding

1. Import the data set in Python.
2. View the dataset
3. See the structure and the summary of the dataset to understand the data.
4. Find out the number of:
	     a.) Numeric attributes:
	     b.) Categorical attributes:Data Understanding

1. Import the data set in Python.
# In[ ]:


import pandas as pd
import numpy as np

mel_housing_data = pd.read_csv("../input/Melbourne_housing_FULL.csv")


# In[ ]:


# 2. View the dataset

mel_housing_data.head()   # first five records


# In[ ]:


# number of rows and colums

mel_housing_data.shape   


# In[ ]:


# 3. See the structure and the summary of the dataset to understand the data.

mel_housing_data.info()  # view data types

4. Find out the number of:
	     a.) Numeric attributes:
	     b.) Categorical attributes:
# In[ ]:


# object attributes will be considered as categorical variables. 

# identify object columns

object_att = mel_housing_data.select_dtypes(['object']).columns
object_att


# In[ ]:


# convert object to categorical attributes

for objects in object_att:
    mel_housing_data[objects] = mel_housing_data[objects].astype('category')  
    
# postcode is float type but it should be categorical

mel_housing_data['Postcode'] = mel_housing_data["Postcode"].astype('category')

mel_housing_data.info()   # objects variables has been identified as categorical attributes.


# ### Data Preparation : Data Cleaning
Duplicate values: Identify if the datasets have duplicate values or not and remove the duplicate values. 

Find out the number of rows present in the dataset
    Before removing duplicate values
    After removing duplicate values
# In[ ]:


# Before removing duplicate column values

mel_housing_data.shape


# In[ ]:


# to remove duplicate rows for all columns.

mel_housing_data.drop_duplicates(keep=False,inplace=True)

#After removing duplicate values

mel_housing_data.shape


# In[ ]:


# looking for duplicate columns
# by looking at the data dictionary,Rooms and bedrooms2 must be having similar values. Need to check for duplicacy.

mel_housing_data['Rooms v Bedroom2'] = mel_housing_data['Rooms'] - mel_housing_data['Bedroom2']
mel_housing_data['Rooms v Bedroom2'].value_counts()

# frequency of 0 difference in Rooms and Bedroom2 is maximum, we shall take this as duplicate.  need to remove it


# In[ ]:


# drop columns  "Bedroom2" and "Rooms v Bedroom2"

mel_housing_data = mel_housing_data.drop(['Bedroom2','Rooms v Bedroom2'],1)

# After removing duplicate values
mel_housing_data.shape

Variable type: Check if all the variables have the correct variable type, based on the data dictionary. If not, then change them.
# In[ ]:


mel_housing_data.info()

# by looking at the data types for all the attributes.  
# 'Data' should be converted into DateTime object. 


# In[ ]:


# convert Date to datetime object.

mel_housing_data['Date'] = pd.to_datetime(mel_housing_data['Date'])
mel_housing_data.info()

For how many attributes did you need to change the data type?
    We changed object type of Date attribute to datetime object.  Missing value treatment: Check which variables have missing values and use appropriate treatments. 
        For each of the variables, find the number of missing values and provide the value that they have been imputed with.
# In[ ]:


# missing value against each columns.  

total_missing = mel_housing_data.isnull().sum().sort_values(ascending=False)
total_missing_df = pd.DataFrame(total_missing)
percentage_miss = (mel_housing_data.isnull().sum()/mel_housing_data.isnull().count()*100).sort_values(ascending=False)
missing_data = pd.concat([total_missing_df, percentage_miss], axis=1, keys=['total_missing_df', 'percentage_miss'])
#missing_data
missing_data_withoutnan = missing_data[missing_data > 0].dropna()
missing_data_withoutnan


# In[ ]:


# total_missing_index = pd.DataFrame(missing_data_withoutnan.index)
# total_missing_index[0][1]

# for cat in total_missing_index[0]:
#     #print (cat)
#     cats = pd.Series(cat)
#     print (mel_housing_data[cats].mean())
#     mel_housing_data[cats].fillna(mel_housing_data[cats].mean(), inplace=True)


# In[ ]:


# dropping the missing data.

mel_housing_data = mel_housing_data.dropna()
mel_housing_data.shape

Outlier Treatment: 
		Identify the variables : Make a subset of the dataset with all the numeric variables. 
		Outliers : For each variable of this subset, carry out the outlier detection. Find out the percentile distribution of each variable and carry out capping and flooring for outlier values.  
# In[ ]:


mel_housing_data.describe()

Identify the variables : Make a subset of the dataset with all the numeric variables.

# In[ ]:



# we can exclude all the attributes related to category and datetime
mel_housing_subset = mel_housing_data.select_dtypes(exclude=['category','datetime64[ns]'])

# as per statistical analysis,  SD for latitude and longitude are close to 0.  It means all the values
# contributing to mean are identical.  so, there won't be any outliers.  we can exclude them. 

mel_housing_subset = mel_housing_subset.drop(['Lattitude','Longtitude'],axis=1)
mel_housing_subset


# In[ ]:


mel_housing_subset.shape

Outliers : For each variable of this subset, carry out the outlier detection. Find out the percentile distribution of each variable and carry out capping and flooring for outlier values.  
# In[ ]:


# describe() will output the percentile distribution of each column

mel_housing_subset.describe()


# In[ ]:


# capping and flooring of 5% on every columns.

import pandas as pd
from pandas import Series

from scipy.stats import mstats
get_ipython().run_line_magic('matplotlib', 'inline')

# Truncate values to the 5th and 95th percentiles
Rooms = pd.Series(mstats.winsorize(mel_housing_subset['Rooms'], limits=[0.05, 0.05])) 
Price = pd.Series(mstats.winsorize(mel_housing_subset['Price'], limits=[0.05, 0.05])) 
Distance = pd.Series(mstats.winsorize(mel_housing_subset['Distance'], limits=[0.05, 0.05])) 
Bathroom = pd.Series(mstats.winsorize(mel_housing_subset['Bathroom'], limits=[0.05, 0.05])) 
Car = pd.Series(mstats.winsorize(mel_housing_subset['Car'], limits=[0.05, 0.05])) 
Landsize = pd.Series(mstats.winsorize(mel_housing_subset['Landsize'], limits=[0.05, 0.05])) 
BuildingArea = pd.Series(mstats.winsorize(mel_housing_subset['BuildingArea'], limits=[0.05, 0.05])) 
YearBuilt = pd.Series(mstats.winsorize(mel_housing_subset['YearBuilt'], limits=[0.05, 0.05])) 
Propertycount = pd.Series(mstats.winsorize(mel_housing_subset['Propertycount'], limits=[0.05, 0.05])) 

mel_housing_subset= pd.concat([Rooms, Price, Distance, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Propertycount], axis=1, keys =['Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Propertycount'])
mel_housing_subset


# ### Data Preparation Feature Engineering:
Feature Transformation:
		Identify variables that have non-linear trends.
		How many variables have non-linear trends?
		Transform them (as required)
# In[ ]:


# import seaborn as sns
# %matplotlib inline
# sns.boxplot(x=mel_housing_subset['Rooms'])


# In[ ]:


# from scipy import stats
# import numpy as np
# z = np.abs(stats.zscore(mel_housing_subset))
# print(z)

# mel_housing_subset = mel_housing_subset[(z < 3).all(axis=1)]
# mel_housing_subset.shape

