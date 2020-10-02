#!/usr/bin/env python
# coding: utf-8

# # Used Car Price Analysis
# The following data is sourced from [Austin Reese's Used Car Data Set on Kaggle.com](https://www.kaggle.com/austinreese/craigslist-carstrucks-data). In the course of this notebook, we will present an analysis of important factors in used car pricing, and then present a basic machine learning model to predict car prices. 

# In[ ]:


#Import necessary packages for data analysis, visualization, and machine learning
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from learntools.core import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Basic Exploratory Data Analysis and Cleaning
# Let's preview the data set:

# In[ ]:


fullCarData = pd.read_csv('../input/craigslist-carstrucks-data/craigslistVehicles.csv')


# In[ ]:


fullCarData.head()


# There are several columns here that aren't particularly useful for modeling or analysis purposes, so I will drop them from the data frame. 

# In[ ]:


carData = fullCarData.drop(labels = ['url', 'city_url', 'VIN', 'image_url', 'desc'],inplace = False, axis = 1)


# In[ ]:


carData.head()


# Let's look at the distribution of the price data:

# In[ ]:


plt.figure(figsize = (15, 10))
plt.title('Distribution of prices')
sns.boxplot(y = carData['price'])


# This is a ridiculous range, we have some very extreme price outliers. Let's try and zoom in on the actual box plot:

# In[ ]:


plt.figure(figsize = (15, 10))
plt.title('Distribution of prices')
plt.ylim(0,50000)
sns.boxplot(y = carData['price'])


# I'm going to eliminate some of the more serious outliers in price by removing entries where the price is over 40000 USD.

# In[ ]:


carData2 = carData.loc[carData['price'] <= 40000]
plt.figure(figsize = (15, 10))
plt.title('Distribution of prices')
plt.ylim(0,50000)
sns.boxplot(y = carData2['price'])


# There are already several NaN entries in the head of this data frame. Let's check to see how many null values there are in each column. Then, we will known whether it's advisable to drop the column or use an imputer.

# In[ ]:


carData2.isna().sum()


# In[ ]:


carData2.count() #Total non-NA values in each column 


# In[ ]:


percentNA = (carData2.isna().sum() / (carData2.isna().sum() + carData2.count())) * 100 #Percent of data missing in each column
print(percentNA)


# The condition, cylinders, odometer, drive, size, type, and paint color columns seem to be missing significant amounts of data. More than half of the Craigs List ads have no information on size, so I will drop that column. 
# 
# Next, let's look at the cardinality of some of these categorical variables.

# In[ ]:


#Find unique number of categories for each categorical variable
print('City: ', carData2['city'].nunique())
print('Year: ', carData2['year'].nunique())
print('Manufacturer: ', carData2['manufacturer'].nunique())
print('Make: ', carData2['make'].nunique())
print('Condition: ', carData2['condition'].nunique())
print('Cylinders: ', carData2['cylinders'].nunique())
print('Fuel: ', carData2['fuel'].nunique())
print('Title Status: ', carData2['title_status'].nunique())
print('Transmission: ', carData2['transmission'].nunique())
print('Drive: ', carData2['drive'].nunique())
print('Size: ', carData2['size'].nunique())
print('Type: ', carData2['type'].nunique())
print('Paint Color: ', carData2['paint_color'].nunique())


# The cardinality of these variables is important because it affects processing time and readability of my visualizations, and high cardinality variables could take a lot of space for very little gain in a machine learning model. 
# 
# It might also help to zero in on a specific range of years to make our data more manageable. I will investigate the data concerning cars from 1990 and later. 

# In[ ]:


subsetData = carData2.loc[carData.year >= 1990]
subsetData.tail()
subsetData.shape


# In[ ]:


print('Manufacturer: ', subsetData['manufacturer'].nunique())
print('Make: ', subsetData['make'].nunique())
print('Condition: ', subsetData['condition'].nunique())
print('Cylinders: ', subsetData['cylinders'].nunique())
print('Fuel: ', subsetData['fuel'].nunique())
print('Title Status: ', subsetData['title_status'].nunique())
print('Transmission: ', subsetData['transmission'].nunique())
print('Drive: ', subsetData['drive'].nunique())
print('Size: ', subsetData['size'].nunique())
print('Type: ', subsetData['type'].nunique())
print('Paint Color: ', subsetData['paint_color'].nunique())


# It's quite clear that the number of makes is a bit unwieldly. Let's pursue other variables for now. 

# Let's look at manufacturer vs price. We should check how many times each manufacturer occurs in the data and set limits on which manufacturers we should consider. 

# In[ ]:


#Remove NAs
manufacturerData = subsetData.loc[subsetData['manufacturer'].isna() == False]

#Count instances of each manufacturer
valueSeries = manufacturerData['manufacturer'].value_counts()
print(valueSeries)


# We have over 500,000 non-NA values for manufacturer, so I will eliminate any manufacturer with fewer than 1000 sales from our consideration.

# In[ ]:


to_remove = valueSeries[valueSeries < 1000].index
manufacturerData['manufacturer'].replace(to_remove, np.nan, inplace = True)


# In[ ]:


manufacturerData['manufacturer'].value_counts()
manufacturerData.loc[manufacturerData['manufacturer'].isna() == False] #We added some nas back in above


# In[ ]:


plt.figure(figsize = (25, 15))
plt.title('Manufacturer vs Price of Used Cars from Model Years 1990 and Above')
g = sns.boxplot(x = manufacturerData['manufacturer'], y = manufacturerData['price'])
g.set_xticklabels(labels = manufacturerData['manufacturer'].unique(),rotation = 90)


# Ram clearly has the highest median resale price. Rover also has a fairly high median resale price. Pontiac, Saturn, and Mercury have fairly low resale prices. The rest of the manufacturers' median resale prices seem to fluctuate around $10000, +/- $5000 . Ram, GMC, and Rover seem to also have larger spreads than some of the other manufacturers' resale prices. 
# 
# Next, let's consider vehicle resale condition vs price. 

# In[ ]:


#Remove NAs for now
conditionData = subsetData.loc[subsetData['condition'].isna() == False]
conditionData.head()


# In[ ]:


#Count instances of each condition
conditionData['condition'].value_counts()


# There are about 300000 condition entries that are non-NA, so I will drop the cars that are in salvage condition because there are fewer than 1000 instances. 

# In[ ]:


conditionData2 = conditionData[conditionData['condition'] !='salvage']
conditionData2['condition'].value_counts()


# In[ ]:


plt.figure(figsize = (15, 10))
plt.title('Condition vs Price of Used Cars from Model Years 1990 and Above')
sns.boxplot(x = conditionData2['condition'], y = conditionData2['price'])


# Interestingly enough, the median price of a like new car is higher than that of a new car on Craig's List. Like new has the highest median resale price, followed by excellent and new, good, and fair condition. New condition has the largest spread, followed by like new. 
# 
# Now, let's consider number of cylinders vs price. 

# In[ ]:


#Remove NAs
cylinderData = subsetData.loc[subsetData['cylinders'].isna() == False]
cylinderData.head()


# In[ ]:


#Get value counts for each number of cylinders
valueSeries2 = cylinderData['cylinders'].value_counts()
print(valueSeries2)


# There are over 300000 non-NA values in the cylinder column, so I will drop all cylinder categories that have fewer than 1000 instances. I'll also remove any rows that have the value 'other' because that's an unhelpful catch-all category.

# In[ ]:


to_remove2 = valueSeries2[valueSeries2 < 1000].index
cylinderData['cylinders'].replace(to_remove2, np.nan, inplace = True)
cylinderData2 = cylinderData.loc[(cylinderData['cylinders'].isna() == False) & (cylinderData['cylinders'] != 'other')]


# In[ ]:


cylinderData2['cylinders'].value_counts()


# In[ ]:


plt.figure(figsize = (15, 10))
plt.title('Number of Cylinders vs Price of Used Cars from Model Years 1990 and Above')
plt.ylim(0, 55000)
sns.boxplot(x = cylinderData2['cylinders'], y = cylinderData2['price'])


#  It seems that cars with 8 cylinder engines have the highest resale price, followed by cars with 10 cylinders, 6 cylinders, 4 cylinders, and 5 cylinders. Cars with 8 and 10 cylinders have the largest spread. 
# 
#  Now we'll investigate fuel type vs price. 

# In[ ]:


#Remove NAs
fuelData = subsetData.loc[subsetData['fuel'].isna() == False]
fuelData.head()


# In[ ]:


fuelData['fuel'].value_counts()


# Again, we have over 500000 non-NA data points for fuel type, so I will remove from consideration fuel types that are represented fewer than 1000 times. We'll also eliminate the 'other' category because it is a catch-all and isn't particularly enlightening.

# In[ ]:


fuelData2 = fuelData.loc[(fuelData['fuel'] != 'electric') & (fuelData['fuel'] != 'other') ]


# In[ ]:


plt.figure(figsize = (15, 10))
plt.title('Fuel Type vs Price of Used Cars from Model Years 1990 and Above')
plt.ylim(0, 70000)
sns.boxplot(x = fuelData2['fuel'], y = fuelData2['price'])


# Diesel vehicles have a much higher median resale price compared to both hybrid and gas vehicles, and also a larger spread. This is interesting!
# 
# Now, we'll consider title status vs price. 

# In[ ]:


#Remove NAs
titleData = subsetData[subsetData['title_status'].isna() == False]


# In[ ]:


valueSeries3 = titleData['title_status'].value_counts()
print(valueSeries3)


# Again, we have over 500000 non-NA values for this column, so I will eliminate any values with fewer than 1000 occurrences.

# In[ ]:


to_remove3 = valueSeries3[valueSeries3 < 1000].index
titleData['title_status'].replace(to_remove3, np.nan, inplace = True)
titleData2 = titleData.loc[titleData['title_status'].isna() == False]
titleData2['title_status'].value_counts()


# In[ ]:


plt.figure(figsize = (15, 10))
plt.title('Title Status vs Price of Used Cars from Model Years 1990 and Above')
plt.ylim(0, 55000)
sns.boxplot(x = titleData2['title_status'], y = titleData2['price'])


# It's a bit surprising to me that the median resale price is higher for cars with the title under lien. This also has a fairly large spread. Salvage, rebuilt, and clean titles are fairly close in median price, although a clean title has a larger spread in resale price. 
# 
# Now we will look at transmission vs price. 

# In[ ]:


#Remove NAs
transmissionData = subsetData.loc[subsetData['transmission'].isna() == False]
transmissionData.head()


# In[ ]:


transmissionData['transmission'].value_counts()


# In[ ]:


#Remove "other" category
transmissionData2 = transmissionData[transmissionData['transmission'] != 'other']
#Plot
plt.figure(figsize = (15, 10))
plt.title('Transmission Type vs Price of Used Cars from Model Years 1990 and Above')
plt.ylim(0, 40000)
sns.boxplot(x = transmissionData2['transmission'], y = transmissionData2['price'])


# It seems that the median resale price of automatic vehicles is only slightly higher than manual vehicles. The automatic vehicle resale prices seem to have a more pronounced skew towards the higher prices, and a greater spread. '
# 
# Next, we will investigate type of drive. 

# In[ ]:


#Remove NAs
driveData = subsetData.loc[subsetData['drive'].isna() == False]
driveData.head()


# In[ ]:


driveData['drive'].value_counts()


# In[ ]:


#Plot
plt.figure(figsize = (15, 10))
plt.title('Drive Type vs Price of Used Cars from Model Years 1990 and Above')
plt.ylim(0, 55000)
sns.boxplot(x = driveData['drive'], y = driveData['price'])


# Four wheel drive has the highest median resale price, followed by rear wheel drive and front wheel drive (in that order). The spread of resale prices for each type follow the same pattern- four wheel drive has the greatest spread, followed by rear wheel drive and front wheel drive. 
# 
# Next, we will consider size vs price. 

# In[ ]:


#Remove NAs
sizeData = subsetData.loc[subsetData['size'].isna() == False]
sizeData.head()


# In[ ]:


sizeData['size'].value_counts()


# In[ ]:


#Plot
plt.figure(figsize = (15, 10))
plt.title('Size vs Price of Used Cars from Model Years 1990 and Above')
plt.ylim(0, 40000)
sns.boxplot(x = sizeData['size'], y = sizeData['price'])


# The differences between mid-size, compact, and sub-compact median prices and spread is minimal. Full-size cars, however, have a higher median resale price and a larger spread. 
# 
# Now, we'll consider vehicle type.

# In[ ]:


#Remove NAs
typeData = subsetData.loc[subsetData['type'].isna() == False]
typeData.head()


# In[ ]:


valueSeries4 = typeData['type'].value_counts()
print(valueSeries4)


# Once again, we have over 300000 non-NA entries in this column, so I will eliminate categories that have fewer than 1000 instances. 

# In[ ]:


to_remove4 = valueSeries4[valueSeries4 < 1000].index
typeData['type'].replace(to_remove4, np.nan, inplace = True)
typeData2 = typeData.loc[(typeData['type'].isna() == False) & (typeData['type'] != 'other')] #also remove 'other' category
typeData2['type'].value_counts()


# In[ ]:


#Plot
plt.figure(figsize = (15, 10))
plt.title('Vehicle Type vs Price from Model Years 1990 and Above')
sns.boxplot(x = typeData2['type'], y = typeData2['price'])


# The highest median resale price belongs to pickup trucks, followed by trucks. Coupes, vans, and SUVS have lower median resale prices than the trucks. Convertibles, wagons, and coupes have the next highest median resale price, followed by mini-vans, sedans, and hatchbacks. pickup trucks and trucks also have the largest spread of resale prices. 
# 
# Now, we will look at paint color vs price. 

# In[ ]:


#Remove NAs
colorData = subsetData.loc[subsetData['paint_color'].isna() == False]
colorData.head()


# In[ ]:


colorData['paint_color'].value_counts()


# Once again, I will get rid of 'purple' because there are fewer than 1000 entries. 

# In[ ]:


colorData2 = colorData[colorData['paint_color'] != 'purple']

#Plot
plt.figure(figsize = (15, 10))
plt.title('Paint Color vs Price from Model Years 1990 and Above')
sns.boxplot(x = colorData2['paint_color'], y = colorData2['price'])


# Notice that the price doesn't vary very much ($10000 +/- $2000) between the color categories. One exception seems to be green cars, which have the lowest median resale price. 
# 
# Now, let's consider odometer readings vs price.
# 

# In[ ]:


#Remove NAs
odometerData = subsetData.loc[subsetData['odometer'].isna() == False]
odometerData.head()

#Plot
plt.figure(figsize = (15, 10))
plt.title('Odometer Reading vs Price from Model Years 1990 and Above')
plt.xlim(0, 1000000)
sns.scatterplot(x = odometerData['odometer'], y = odometerData['price'])


# As predicted, there seems to be a strong negative correlation between price and mileage.

# # Predicting Resale Price using Machine Learning
# We're going to use gradient boosting to try and create a machine learning model to predict the prices of these cars.

# In[ ]:


#Drop any rows where the price is NA
carData2.dropna(axis = 0, subset = ['price'], inplace = True)


# In[ ]:


#Separate price data from rest of table, remove some columns from X I don't care about
y = carData2.price
X = carData2.drop(labels = ['price'], inplace = False, axis = 1)


# In[ ]:


list(X) #Check that price isn't there


# In[ ]:


y.head() #Take a look at y


# Now, we're going to split our data further into training and validation sets. 

# In[ ]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2)


# Now, we will separate the variables by type (categorical vs numerical) so that we can preprocess the data effectively. I'm also eliminating some columns that have too high a cardinality.

# In[ ]:


categorical_cols = [cname for cname in X_train_full.columns if (X_train_full[cname].dtype == 'object') and (X_train_full[cname].nunique() <= 10)]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[ ]:


print(categorical_cols)


# In[ ]:


print(numerical_cols)


# In[ ]:


#Preprocess data
numerical_transformer = SimpleImputer(strategy = 'most_frequent')
categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')),
                                            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
                                            ])
preprocessor = ColumnTransformer(transformers = [('num', numerical_transformer, numerical_cols),
                                                ('cat', categorical_transformer, categorical_cols)
                                                ])


# Now, we will create a pipeline, fit it to the training data, and make predictions. Then, we will look at the mean absolute error to evaluate the model. 

# In[ ]:


my_pipeline = Pipeline(steps = [("preprocessor", preprocessor), ("xgbrg", XGBRegressor(n_estimators = 1000, 
                          learning_rate = .05))])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)


# This is improved over some other results I've gotten, but we still have some parameter tuning to do.

# In[ ]:





# In[ ]:




