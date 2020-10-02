#!/usr/bin/env python
# coding: utf-8

# # Melbourne Housing Market: Multiple Linear Regression

# EDA - Visualization - Modeling

# <img src="https://i.imgur.com/dXIK3yX.jpg" width="600px">

# Photo by [Breno Assis](https://unsplash.com/photos/r3WAWU5Fi5Q) on Unsplash

# ## 1. Dataset

# https://www.kaggle.com/anthonypino/melbourne-housing-market

# * Suburb: Suburb
# * Address: Address
# * Rooms: Number of rooms
# * Price: Price in Australian dollars
# * Method: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.
# * Type: br - bedroom(s); h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; dev site - development site; o res - other residential.
# * SellerG: Real Estate Agent
# * Date: Date sold
# * Distance: Distance from CBD in Kilometres
# * Regionname: General Region (West, North West, North, North east ...etc)
# * Propertycount: Number of properties that exist in the suburb.
# * Bedroom2 : Scraped # of Bedrooms (from different source)
# * Bathroom: Number of Bathrooms
# * Car: Number of carspots
# * Landsize: Land Size in Metres
# * BuildingArea: Building Size in Metres
# * YearBuilt: Year the house was built
# * CouncilArea: Governing council for the area
# * Lattitude: Self explanitory
# * Longtitude: Self explanitory

# Dataset downloaded from Kaggle on the 2020/01/05

# ## 2. Exploratory Data Analysis

# ### 2.1 Explore the dataset structure

# In[ ]:


# Version v01-02
# Import all libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt # ploting the data
import seaborn as sns # ploting the data
import math # calculation


# In[ ]:


# Set up color blind friendly color palette
# The palette with grey:
cbPalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
# The palette with black:
cbbPalette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# sns.palplot(sns.color_palette(cbPalette))
# sns.palplot(sns.color_palette(cbbPalette))

sns.set_palette(cbPalette)
#sns.set_palette(cbbPalette)


# In[ ]:


# Load the dataset
#price_less = pd.read_csv('Data/MELBOURNE_HOUSE_PRICES_LESS.csv')
price_less = pd.read_csv('../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv')
# price_full = pd.read_csv('Data/Melbourne_housing_FULL.csv')


# In[ ]:


price_less.info()


# In[ ]:


# price_full.info()


# In[ ]:


price_less.head(10)


# In[ ]:


# price_full.head(10)


# In[ ]:


# Determine the number of missing values for every column
price_less.isnull().sum()


# In[ ]:


# Exclude rows with missing prices
data_filtered = price_less.loc[price_less['Price'] > 0]
data_filtered.isnull().sum()


# In[ ]:


# price_full.isnull().sum()


# ### 2.2 Explore the continous variables

# In[ ]:


data = data_filtered.copy()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


data['Price'].describe()


# In[ ]:


x = 'Price'
sns.set_palette("muted")
sns.distplot(data[x])
plt.ioff()
sns.set_palette(cbPalette)


# In[ ]:


# Log transform the Price variable to approach a normal distribution
x = np.log10(data["Price"])
sns.set_palette("muted")
sns.distplot(x)
plt.ioff()
sns.set_palette(cbPalette)


# In[ ]:


# data["Price"] = np.log10(data.loc[:, "Price"].values)


# In[ ]:


# data["Price"] = np.log10(data.loc[:, "Price"].values)


# In[ ]:


x = 'Rooms'
sns.set_palette("muted")
sns.distplot(data[x])
plt.ioff()
sns.set_palette(cbPalette)


# In[ ]:


x = 'Propertycount'
sns.set_palette("muted")
sns.distplot(data[x])
plt.ioff()
sns.set_palette(cbPalette)


# In[ ]:


x = 'Distance'
sns.set_palette("muted")
sns.distplot(data[x])
plt.ioff()
sns.set_palette(cbPalette)


# ### 2.1 Explore the categorical variables

# In[ ]:


data.head()


# In[ ]:


# data.shape


# In[ ]:


# data.columns


# In[ ]:


#  https://www.datacamp.com/community/tutorials/categorical-data
data['Suburb'].value_counts().count()


# In[ ]:


#  https://www.datacamp.com/community/tutorials/categorical-data
data['Address'].value_counts().count()


# In[ ]:


# https://seaborn.pydata.org/generated/seaborn.countplot.html
title = 'Count of properties per Region'
sns.countplot(y = data['Regionname'])
plt.title(title)
plt.ioff()


# In[ ]:


# https://seaborn.pydata.org/generated/seaborn.countplot.html
title = ''
sns.countplot(y = data['Type'])
plt.title(title)
plt.ioff()


# In[ ]:


#  https://www.datacamp.com/community/tutorials/categorical-data
# data['Method'].value_counts()


# In[ ]:


title = ''
sns.countplot(y = data['Method'])
plt.title(title)
plt.ioff()


# In[ ]:


#  https://www.datacamp.com/community/tutorials/categorical-data
# price_less['CouncilArea'].value_counts().count()


# In[ ]:


# price_less['CouncilArea'].value_counts()


# In[ ]:


title = ''
plt.figure(figsize=(20,10))
sns.countplot(y = price_less['CouncilArea'])
plt.title(title)
plt.ioff()


# In[ ]:


# title = ''
# plt.figure(figsize=(20,10))
# sns.countplot(y = price_less['SellerG'])
# plt.title(title)
# plt.ioff()


# In[ ]:


data['SellerG'].value_counts().count()


# In[ ]:


data['Postcode'].value_counts().count()


# ## 3. Visualization

# ## 3.1 Price vs continous variables

# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


# see https://seaborn.pydata.org/generated/seaborn.scatterplot.html
sns.set_palette("muted")
x = 'Rooms'
# x = np.log10(data["Rooms"])
# y = 'Price'
y = np.log10(data["Price"])

title = ''
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=data)
plt.title(title)
plt.ioff()
sns.set_palette(cbPalette)


# In[ ]:


sns.set_palette("muted")
x = "Distance"
y = np.log10(data["Price"])

title = ''
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=data)
plt.title(title)
plt.ioff()
sns.set_palette(cbPalette)


# In[ ]:


sns.set_palette("muted")
x = np.log10(data["Price"] / data["Rooms"])
y = np.log10(data["Price"])

title = ''
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=data)
plt.title(title)
plt.ioff()
sns.set_palette(cbPalette)


# In[ ]:


sns.set_palette("muted")
x = np.log10(data['Propertycount'])
# x = np.log10(data['Propertycount'] / data["Rooms"])
y = np.log10(data["Price"])

title = ''
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=x, y=y, data=data)
plt.title(title)
plt.ioff()
sns.set_palette(cbPalette)


# ## 3.2 Price vs categorical variables

# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


y="Type"
x=np.log10(data["Price"])

title = ""
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=x, y=y, data=data, notch=True, showmeans=True,
           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
plt.title(title)
plt.ioff()


# In[ ]:


y="Regionname"
x=np.log10(data["Price"])
# x="Price"

title = ""
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=x, y=y, data=data, notch=True, showmeans=True,
           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
plt.title(title)
plt.ioff()


# In[ ]:


y="Method"
x=np.log10(data["Price"])
# x="Price"

title = ""
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=x, y=y, data=data, notch=True, showmeans=True,
           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
plt.title(title)
plt.ioff()


# In[ ]:


sns.set_palette("muted")
y="CouncilArea"
x=np.log10(data["Price"])
# x="Price"

title = ""
f, ax = plt.subplots(figsize=(25, 10))
sns.boxplot(x=x, y=y, data=data, notch=False, showmeans=True,
           meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
plt.title(title)
plt.ioff()
sns.set_palette(cbPalette)


# ## 4. Models

# ### 4.1 Data Preprocessing

# In[ ]:


# https://stackoverflow.com/questions/31468176/setting-values-on-a-copy-of-a-slice-from-a-dataframe
data['Price/Rooms'] = (data.loc[:, "Price"] / data.loc[:, "Rooms"])


# In[ ]:


data.columns


# In[ ]:


data.drop(['Date', 'Address'], axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


# Encoding categorical data
# https://pbpython.com/categorical-encoding.html
data = pd.get_dummies(data, columns=['Suburb', 'Rooms', 'Type',  'Method', 'SellerG', 'Regionname', 'CouncilArea'], drop_first=True)


# In[ ]:


data.info()


# In[ ]:


# Split the dataset
X = data.drop('Price', axis=1).values
y = data['Price'].values
y = np.log10(y)


# ### 4.2 Multiple Linear Regression

# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lr.predict(X_test)


# In[ ]:


# Compare predicted and actual values
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# https://stackoverflow.com/questions/19100540/rounding-entries-in-a-pandas-dafaframe
df = pd.DataFrame({'Actual': np.round(y_test, 2), 
                   'Predicted': np.round(y_pred, 2)})
df.head(10)


# In[ ]:


# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score

print('Price mean:', np.round(np.mean(y), 2))  
print('Price std:', np.round(np.std(y), 2))
print('RMSE:', np.round(np.sqrt(metrics.mean_squared_error(y_test, lr.predict(X_test))), 2))
print('R2 score train:', np.round(r2_score(y_train, lr.predict(X_train), multioutput='variance_weighted'), 2))
print('R2 score test:', np.round(r2_score(y_test, lr.predict(X_test), multioutput='variance_weighted'), 2))


# ## 5. Conclusions
# * Price is clearly proportional to the Distance and Rooms number variables
# * Multiple linear regression performs well on this dataset

# ## 6. References

# * https://www.kaggle.com/lpuglisi/visualizing-melbourne-real-estate 
# * https://www.kaggle.com/anthonypino/price-analysis-and-linear-regression
# * https://www.kaggle.com/emanueleamcappella/random-forest-hyperparameters-tunings
# * https://www.datacamp.com/community/tutorials/categorical-data
# * https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f
# * https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/
# * https://pbpython.com/categorical-encoding.html
