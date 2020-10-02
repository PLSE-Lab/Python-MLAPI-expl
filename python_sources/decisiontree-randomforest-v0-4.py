#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Data Set and EDA
# We will use Housing Price dataset to apply first three ML models. This data set has several features related to respective houses. SalesPrice is the target variable (y). We can use rest of the features as Independent variables (x).

# In[ ]:


# Path of the file to read
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
home_data1 = home_data[home_data.SalePrice<400000]


# In[105]:


home_data.head()


# In[109]:


# home_data.SalePrice.hist(bins=100)
home_data1.SalePrice.hist(bins=100)


# The above graph shows the data is right skewed.

# ## **Implement Decision Tree**

# In[ ]:


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


# Create target object and call it y

y = home_data1.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data1[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(criterion='mse', random_state=1,max_leaf_nodes=100)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))


# In[ ]:


## Write a function to find out the best value of max leaf node


# ## Model Explainability
# Let's visualize the model.

# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(iowa_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ## What is the feature importance as per model

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
importances = iowa_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [features[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()


# ## Error Analysis

# In[ ]:


data=pd.DataFrame({'pred':val_predictions,'act':val_y}).sort_index(ascending=True)
data['pred']=data['pred'].astype('int')
data['err']=data.act-data.pred


# In[ ]:


import matplotlib.pyplot as plt
fig, axarr = plt.subplots(figsize=(18, 8))

final=data.sort_values('act',ascending=False)
final = final[['act','err']]
final.plot.bar(stacked=True,ax=axarr)


# In[ ]:


data.plot.scatter('pred','act')


# ## **Implement Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1,criterion='mse',min_samples_leaf=4)

# fit your model
rf_model.fit(train_X,train_y)
val_pred = rf_model.predict(val_X)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_pred,val_y)

print("Validation MAE for Random Forest Model: {:.0f}".format(rf_val_mae))


# In[ ]:


importances = rf_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [features[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()


# In[ ]:


data2=pd.DataFrame({'pred':val_pred,'act':val_y}).sort_index(ascending=True)
data2['pred']=data2['pred'].astype('int')
data2['err']=data2.act-data2.pred


# In[ ]:


import matplotlib.pyplot as plt
fig, axarr = plt.subplots(figsize=(18, 8))

final2=data2.sort_values('act',ascending=False)
final2 = final2[['act','err']]
final2.plot.bar(stacked=True,ax=axarr)


# In[ ]:


data2.plot.scatter('pred','act')


# ## **Implement Random Forest with All other variables**
# First find out variables that have too many missing values. Drop such variables.
# 
# Fill the missing values with 0.
# 
# Convert the data with one hot encoding

# In[ ]:


a= home_data.isna().mean().round(4) * 100
a[a.values>0]


# In[102]:


y = home_data.SalePrice
X = home_data.iloc[:,:-1]
X = X.drop(columns=['Alley','Fence','FireplaceQu','MiscFeature','PoolQC'])
X = X.fillna(0)
X = pd.get_dummies(X)
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# In[103]:


rf_model = RandomForestRegressor(random_state=1,criterion='mse',min_samples_leaf=4)

# fit your model
rf_model.fit(train_X,train_y)
val_pred = rf_model.predict(val_X)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_pred,val_y)

print("Validation MAE for Random Forest Model: {:.0f}".format(rf_val_mae))


# In[104]:


features = X.columns.values
importances = rf_model.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [features[i] for i in indices]

# Create plot
plt.figure(figsize=(50,10))

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()


# In[106]:


data3=pd.DataFrame({'pred':val_pred,'act':val_y}).sort_index(ascending=True)
data3['pred']=data3['pred'].astype('int')
data3['err']=data3.act-data3.pred


# In[107]:


import matplotlib.pyplot as plt
fig, axarr = plt.subplots(figsize=(18, 8))

final3=data3.sort_values('act',ascending=False)
final3 = final3[['act','err']]
final3.plot.bar(stacked=True,ax=axarr)


# In[108]:


data3.plot.scatter('pred','act')


# ## Dataset Fields description
# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: $Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale
