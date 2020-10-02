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


# # Data Pre-processing
# * import libraries
# * read data
# * data info, shape, describe, head, dtypes
# * convert object to category except date
# * convert date to datetime
# * convert numeric (postcode) to category
# * remove duplicates
# * remove columns not informative

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn


# In[ ]:


df_org = pd.read_csv('/kaggle/input/melbourne-housing-market/Melbourne_housing_FULL.csv')
df = pd.read_csv('/kaggle/input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv')


# In[ ]:


df_org.head(3)


# In[ ]:


df.head(3)


# In[ ]:


df_org.columns


# In[ ]:


extra_col = []
for col in df_org.columns:
    if col not in df.columns:
        extra_col.append(col)
print(extra_col)


# In[ ]:


print(df_org.shape)
print(df_org.dtypes)


# In[ ]:


df_org.info()


# In[ ]:


df_org.describe().T


# In[ ]:


print(df_org.select_dtypes(['object']).columns)


# In[ ]:


# convert objects to categorical variables
obj_cats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea',
       'Regionname']
for col in obj_cats:
    df_org[col] = df_org[col].astype('category')


# In[ ]:


# convert date from object to date format
# after converting the date to category, it would not change in the datetime
df_org['Date'] = pd.to_datetime(df_org['Date'])


# In[ ]:


# converting postcode 'numeric variable' to categorical
df_org['Postcode'] = df_org['Postcode'].astype('category')


# Duplicate variables
# 
# 'Rooms' and 'Bedroom2' both contain information on the number of rooms of a home, 

# In[ ]:


# examine rooms vs bedroom2
df_org['Room v Bedroom2'] = df_org['Rooms'] - df_org['Bedroom2']
df_org


# In[ ]:


df_org['Room v Bedroom2'].value_counts()


# The differences between these variables are minimal so keeping both would only be duplicating information. Thus, the Bedroom2 feature will be removed from the data set altogether to allow for better analysis downstream.

# In[ ]:


# Drop columns
df_org = df_org.drop(['Room v Bedroom2', 'Bedroom2'], axis = 1)


# ### Feature Engineering
# The dataset contains the year the home was built. Although this is being measured by the specific year, what this variable is really probing is the age of the home. As such, home age can be expressed in terms of historic(greater than 50 years old) vs non-historic(less than 50 years old) to get the heart of this information in a more condensed way, allowing for better analysis and visualization.

# In[ ]:


# Add age variable
df_org['Age'] = 2017 - df_org['YearBuilt']

# identify historic homes
df_org['Historic'] = np.where(df_org['Age'] >= 50, 'Historic', 'Contemporary')

#convert to category
df_org['Historic'] = df_org['Historic'].astype('category')


# In[ ]:


df_org.Historic.value_counts()


# ### Missing Data
# Based on a quick look in the info() there are some features having missing values. so lets explore that and deal with them. 

# In[ ]:


# Visualize the missing values

fig, ax = plt.subplots(figsize = (15,7))
sns.set(font_scale = 1.2)
sns.heatmap(df_org.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

plt.show()


# In[ ]:


# Count of missing value
df_org.isnull().sum()


# In[ ]:


# percentage of missing values
df_org.isnull().sum()/len(df_org)*100


# There are a significant amount of missing values in Price, Bathroom, Car, Landsize, Building Area, YearBuilt, Council Area, Lattitude, and Longitude. To allow for a more complete analysis, observations missing any data will be removed from the dataset.

# In[ ]:


print(df_org.shape)
# to remove rows missing data in a specific column
# as the yearbuilt column has some missing value hence it would miscalculate the historic col
df_org = df_org[pd.notnull(df_org['YearBuilt'])]
print(df_org.shape)


# In[ ]:


# drop all rows having null vlaues
df_org = df_org.dropna()
print(df_org.shape)


# In[ ]:


plt.figure(figsize = (12, 7))
sns.heatmap(df_org.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# ## Outliers
# The statistical summary revealed minimum values of zero for landsize and buildingArea that seem odd. Also, there is a max price of $8.4 million in the dataset. These observations will need to be investigate further to determne their validity and whether they should be included in the dataset for analysis.
# 

# In[ ]:


df_org.describe().T


# In[ ]:


df_org[df_org['Age']>800]


# In[ ]:


df_org[df_org['BuildingArea'] == 0]


# In[ ]:


df_org[df_org['Landsize'] == 0]


# After additional research, I determined that a zero land size could be indicative of 'zero-lot-line' homes-residential real estate in which the structure comes up to or very near the edge of the property line. Therefore, these observations are valid and will remain in the dataset.
# 
# However, the observations with 'zero' BuildingArea will be removed because it is not possible for a home to have a size of zero. Also, this observation is priced ussually high at $8.4 M(the outlier identified earlier), further confirming a possible error in the data point. For these two reasons, this observations will be removed. 
# Remove outlier
df_org = df_org[df_org['BuildingArea'] != 0]
# In[ ]:


# Confirm removal
df_org[df_org['BuildingArea'] == 0]


# In[ ]:


df_org.describe().T


# # Exploratory Data Analysis (EDA)

# lets visualize the target variable

# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df_org['Price'], kde = False)


# ## Bivariate
# * visualization of categorical Features
#     
#     relationship between target and categorical features(suburb, address, postcode). inspite of using all of these features in the analysis, Regioname would be the best proxy of home location to use for analysis.
#     
#     based on the domain knowledge, home's real estate agent has minimal effect on a price relative to other features and will be excluded from further analysis.

# In[ ]:


#columns having null values
#null_col =df_org.columns[df_org.isnull().any()].tolist()
#null_col


# In[ ]:


#categorical features
df_org.select_dtypes(['category']).columns


# In[ ]:


df_org['Regionname'].value_counts()


# In[ ]:


# Abbreviate Regionname categories
df_org['Regionname'] = df_org['Regionname'].map({'Southern Metropolitan' : 'S Metro',
                                                'Northern Metropolitan' : 'N Metro',
                                                'Western Metropolitan': 'W Metro', 
                                                'Eastern Metropolitan': 'E Metro', 
                                                'South-Eastern Metropolitan': 'SE Metro',
                                                'Northern Victoria': 'N Vic',
                                                'Eastern Victoria': 'E Vic',
                                                'Western Victoria' : 'W Vic'})


# In[ ]:


df_org['Regionname'].value_counts()


# In[ ]:


# Subplot of categorical features vs price
sns.set_style('whitegrid')
f, ax = plt.subplots(2,2, figsize = (15,15))

# plot [0,0]
sns.boxplot(data = df_org, x = 'Type', y = 'Price', ax= ax[0,0])
ax[0,0].set_xlabel('Type')
ax[0,0].set_ylabel('Price')
ax[0,0].set_title('Type vs Price')

# Plot[0,1]
sns.boxplot(data = df_org, x = 'Method', y= 'Price', ax = ax[0,1])
ax[0,1].set_xlabel('Method')
ax[0,1].set_title('Method vs Price')

# Plot[1,0]
sns.boxplot(data = df_org, x = 'Regionname', y= 'Price', ax = ax[1,0])
ax[1,0].set_xlabel('Regionname')
ax[1,0].set_title('Region Name vs Price')

# Plot[1,1]
sns.boxplot(data = df_org, x = 'Historic', y= 'Price', ax = ax[1,1])
ax[1,1].set_xlabel('Historic')
ax[1,1].set_title('Historic vs Price')

plt.show()


# ## Insights
# * Median price for houses are over 1M, townhomes are 800k- 900k and units are approx  500k. 
# * Home prices with different selling methods are relatively the same across the board.
# * Median prices in the Metropolitan Region are higher than that of Victoria Reigon- with Southern metro being the area with the highest median home price (~1.3M).
# * With an average price of 1M, historic homes(older than 50 years old) are valued much higher than newer homes in the area, but have more variation in price. 
# 

# ## Numeric Features
# Now, I visualize the relationships between numeric features in the dataest with price. 

# In[ ]:


# Identify numeric features
df_org.select_dtypes(['float64', 'int64']).columns


# In[ ]:


# subplots of numeric features vs price
sns.set_style('whitegrid')
fig, ax = plt.subplots(4,2, figsize =(30,40))

#plot[0,0]
ax[0,0].scatter(x = 'Rooms', y = 'Price', data = df_org, edgecolor = 'b')
ax[0,0].set_xlabel('Rooms')
ax[0,0].set_ylabel('Price')
ax[0,0].set_title('Rooms vs Price')

#Plot [0,1]
ax[0,1].scatter(x = 'Distance', y = 'Price', data = df_org, edgecolor = 'b')
ax[0,1].set_xlabel('Distance')
ax[0,1].set_ylabel('Price')
ax[0,1].set_title('Distance vs Price')

#Plot [1,0]
ax[1,0].scatter(x = 'Bathroom', y = 'Price', data = df_org, edgecolor = 'b')
ax[1,0].set_xlabel('Bathroom')
ax[1,0].set_ylabel('Price')
ax[1,0].set_title('Bathroom vs Price')

#Plot [1,1]
ax[1,1].scatter(x = 'Car', y = 'Price', data = df_org, edgecolor = 'b')
ax[1,1].set_xlabel('Car')
ax[1,1].set_ylabel('Price')
ax[1,1].set_title('Car vs Price')

#Plot [2,0]
ax[2,0].scatter(x = 'Landsize', y = 'Price', data = df_org, edgecolor = 'b')
ax[2,0].set_xlabel('Landsize')
ax[2,0].set_ylabel('Price')
ax[2,0].set_title('Landsize vs Price')

#Plot [2,1]
ax[2,1].scatter(x = 'BuildingArea', y = 'Price', data = df_org, edgecolor = 'b')
ax[2,1].set_xlabel('BuildingArea')
ax[2,1].set_ylabel('Price')
ax[2,1].set_title('BuildingArea vs Price')

#Plot [3,0]
ax[3,0].scatter(x = 'Age', y = 'Price', data = df_org, edgecolor = 'b')
ax[3,0].set_xlabel('Age')
ax[3,0].set_ylabel('Price')
ax[3,0].set_title('Age vs Price')

#Plot [3,1]
ax[3,1].scatter(x = 'Propertycount', y = 'Price', data = df_org, edgecolor = 'b')
ax[3,1].set_xlabel('Propertycount')
ax[3,1].set_ylabel('Price')
ax[3,1].set_title('Propertycount vs Price')

plt.show()


# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(x= 'Rooms', y = 'Price', data = df_org)


# ## Insights
# * The majority of homes in the dataset have 4, 5 rooms.
# * The most prominent trend is that there is a negative correlation between distance from Melbourne's Central Business District (CBD) and Price. The most expensive homes (2M or more) tend to be within 20Km of the CBD.
# 

# # Correlation
# To explore the correlation of variables with one another.

# In[ ]:


plt.figure(figsize = (10,6))
sns.heatmap(df_org.corr(), cmap = 'coolwarm', linewidth = 1, annot = True, annot_kws = {'size':9})
plt.title('Variable Correlation')


# ### Weak Positive Correlation
# Age and Price
# ### Moderate Positive Correlation
# Rooms and Price
# Bathrooms and Price
# Building Area and Price
# 
# The Rooms, Bathroom and Building Area features are also moderately correlated with one another as they are all measures of home size. 

# # Linear Regression

# In[ ]:


# Identify numeric features
df_org.select_dtypes(['float64', 'int64']).columns


# In[ ]:


# Split test and train
X = df_org[['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize',
       'BuildingArea', 'Propertycount','Age']]
y = df_org['Price']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 0)


# In[ ]:


# model fitting and prediction
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[ ]:


#predictions
y_pred = lm.predict(X_test)


# ### Regression Evaluation Metrics
# Three common evaluation metrics for regression problems:
# 1. Mean Absolute Error (MAE)
# 2. Mean Squared Error (MSE)
# 3. Root Mean Squared Error (RMSE)
# All basic variations on the difference between what you predicted and the true values.
# 
# Comparing these metrics:
# 
# **MAE ** is the easiest to understand, because it's the average error.
# 
# **MSE** more popular than MAE, because MSE "punishes" larger errors, tends to be useful in the real world.
# 
# **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units (target units) . 
# All of these are loss functions, because we want to minimize them.
# 

# In[ ]:


from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# RMSE tells us explicitly how much our predictions deviate, on average, from the actual values in the dataset. 

# In[ ]:


# Calulate R squared
print('R^2 =', metrics.explained_variance_score(y_test, y_pred))


# According to the R-squared, 47.6% of the variance in the dependent variables is explained by the model.

# ## Analyze the Residuals

# In[ ]:


# Actual vs predictions scatter
plt.scatter(y_test, y_pred)


# In[ ]:


# Histogram of the distribution of residuals
sns.distplot((y_test - y_pred))


# ## Interpreting the Coefficients

# In[ ]:


cdf = pd.DataFrame(data = lm.coef_, index = X.columns, columns = ['Coefficients'])
cdf


# # Conclusion

# Every one unit increase in:
# * Rooms is associated with an increase in Price by 136,531.55
# * Distance is associated with a decrease in Price by 32,160.84
# * Bathroom is associated with an increase in Price by 236,639.21
# * Car space is associated with an increase in Price by 59,122.83
# * Landsize is associated with an increase in Price by 35.75
# * BuildingArea is associated with an increase in Price by 26,65.10
# * Propertycount is associated with a decrease in Price by 0.05
# * Age is associated with an increase in Price by 4,729.73
