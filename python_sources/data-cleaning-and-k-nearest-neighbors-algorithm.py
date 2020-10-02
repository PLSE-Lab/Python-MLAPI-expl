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


# In[ ]:


# importing the training data
train_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


# checking the information about the dataset including the data types and numerical counts
train_dataset.info()


# From the information we can see that 5 features 
# 1. LotFrontage (Linear feet of street connected to property) has only 259 values
# 1. Alley (Type of Aleey access) has only 91 values
# 1. FireplaceQU (Quality of Fire Place) has only 770 values
# 1. PoolQC (Quality of the Pool) has only 7 values
# 1. Fence (Quality of the fence) has 281 values
# 1. MiscFeature (Miscellaneous feature not covered in other categories) has 54 values
# 
# *Interences*: The remaining are missing values. **We can remove the columns since they have drastic amount of missing values**.
# 
# The Feature ID is the identity column. so we can consider it as ID column and SalePrice is the Target Feature.

# In[ ]:


# making the Id as index feature
train_dataset = train_dataset.set_index('Id')


# In[ ]:


# checking for missing values
missing_dataframe = pd.concat([train_dataset.isnull().sum()], axis = 1)
print(missing_dataframe[missing_dataframe[0]>0])


# *Intefences*: The remaining columns with missing data can be retained. We can use dropna() option to omit the missing values. Later we can experiment with imputation of some missing values.

# In[ ]:


train_dataset = train_dataset.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)


# In[ ]:


# checking for missing values
missing_dataframe = pd.concat([train_dataset.isnull().sum()], axis = 1)
print(missing_dataframe[missing_dataframe[0]>0])


# In[ ]:


# now we can remove the missing samples from the dataset using dropna()
train_dataset = train_dataset.dropna()


# In[ ]:


# checking for missing values
missing_dataframe = pd.concat([train_dataset.isnull().sum()], axis = 1)
print(missing_dataframe[missing_dataframe[0]>0])


# Removed all the missing values from the dataset. Now we will check for Correlation.
# The best way (from my experience) to check the correlation is to use the heatmap from seaborn library. 
# 
# The features can be positively correlated, negatively correlated or 0 correlated. 

# In[ ]:


numeric_columns = train_dataset.describe().columns
nonnumeric_columns = [col for col in train_dataset.columns if col not in train_dataset.describe().columns]


# In[ ]:


print("numeric columns count: ", len(numeric_columns))
print("non numeric columns count: ", len(nonnumeric_columns))


# Since there are many non numerical columns there is a need to convert them into numerical using Label Encoder or One Hot encoder. 
# One hot encoder creates new features whereas label encoder uses the same features.

# In[ ]:


#import label encoder
from sklearn.preprocessing import LabelEncoder


# In[ ]:


def encoding(dataframe_feature):
    if(dataframe_feature.dtype == 'object'):
        return LabelEncoder().fit_transform(dataframe_feature)
    else:
        return dataframe_feature


# In[ ]:


train_dataset = train_dataset.apply(encoding)


# In[ ]:


numeric_columns = train_dataset.describe().columns
nonnumeric_columns = [col for col in train_dataset.columns if col not in train_dataset.describe().columns]
print("numeric columns count: ", len(numeric_columns))
print("non numeric columns count: ", len(nonnumeric_columns))


# Now all the features are converted into numerical features. Now we can apply regression algorithms and predict the sales price easily.

# In[ ]:


# importing seaborn library for data visualization
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train_dataset.head()


# In[ ]:


sns.heatmap(train_dataset.corr())


# The heatmap is having many features which makes visualization very difficult. 
# 
# *Inferences*: Need to select the top correlated features.

# In[ ]:


correlation_matrix = train_dataset.corr()
essential_features = correlation_matrix.index[abs(correlation_matrix['SalePrice']) > 0.6]
plt.figure(figsize = (8, 8))
sns.heatmap(train_dataset[essential_features].corr(), cbar = False, annot = True, square = True)


# The sale price is positively highly correlated with 
# 1. QverallQual (overall material and finish quality)
# 2. GrLivArea (Above ground living area square feet)
# 3. GarageCars (Size of Garage in car capacity)
# 4. GarageArea (Size of garage in square feet)
# 
# The sale price is negatively highly correlated with
# 1. KitchQual (The Kitchen Quality)
# 2. BsmtQual (The height of the basement)
# 3. ExterQual (The quality of exterior material)
# 
# *Inferences*: If the quality increases the price increases (Obviously). The price is directly depend on area, car parking which means most of the people around that area owns a car or the particular area is well accessible to roadways. 
# 
# By analyzing the negatively correlated features, can we say people dont care about Kitchen, Basement and Exterior Material Quality ? (I dont know)

# Now we will check for outliers. 
# 
# The basic way to check for outliers is by visualizing using scatter plot.
# Since all the features are numerical now we can use seaborn pairplot 

# In[ ]:


sns.pairplot(train_dataset[essential_features])


# The features TotalBsmtSF, 1stFirSF, GrLivArea, GarageArea are having outliers. 

# In[ ]:


train_dataset = train_dataset[(train_dataset["SalePrice"] < 500000) &
              (train_dataset["GrLivArea"] < 3000) &
              (train_dataset["TotalBsmtSF"] < 2300) &
              (train_dataset["1stFlrSF"] < 2200) & 
              (train_dataset["GarageArea"] < 1200)]


# In[ ]:


train_dataset.shape


# In[ ]:


sns.pairplot(train_dataset[essential_features])


# Now the outliers are removed from the dataset. 
# The dataset looks better cleaned now. We shall go into the modelling of data.

# In[ ]:


# making the input and target features
X = train_dataset.drop(['SalePrice'], axis = 1)
y = train_dataset['SalePrice'].values


# In[ ]:


# Splitting the training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


# importing metrics for accuracy calculation
from sklearn.metrics import mean_squared_error


# In[ ]:


# importing KNN regression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
# fitting the model with the training dataset
knn.fit(X_train, y_train)
# predicting the values
predicted_value = knn.predict(X_test)
# calculating the accuracy
rmse_before_cleaning = np.sqrt(mean_squared_error(predicted_value, y_test))
print(rmse_before_cleaning)


# Performances are not upto the expectations. Need to work more.

# In[ ]:


sns.distplot(train_dataset['SalePrice'], bins = 10)


# Inferences - We can see the sale price is positively skewed.

# Using GridSearch CV to find the best parameters of KNN algorithm.

# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = {
            'n_neighbors' : [1,2,3,4,5,6,7,8,9,10],
            'algorithm' : ['ball_tree', 'brute']
             }
grid_search_cv = GridSearchCV(KNeighborsRegressor(), parameters)
grid_search_cv.fit(X_train, y_train)


# In[ ]:


grid_search_cv.best_params_


# Now the dataset has been cleaned and we got the best parameters for the dataset. 

# In[ ]:


knn = KNeighborsRegressor(n_neighbors = 4)
# fitting the model with the training dataset
knn.fit(X_train, y_train)
# predicting the values
predicted_value = knn.predict(X_test)
# calculating the accuracy
rmse_after_cleaning = np.sqrt(mean_squared_error(predicted_value, y_test))
print(rmse_after_cleaning)


# In[ ]:


print('RMSE:',rmse_after_cleaning)


# In[ ]:


# importing test data
test_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_dataset.head()


# In[ ]:


test_dataset = test_dataset.set_index(['Id'])


# In[ ]:


test_dataset.info()


# In[ ]:


test_dataset.isnull().sum()


# In[ ]:


test_dataset = test_dataset.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)


# In[ ]:


test_dataset = test_dataset.dropna()


# In[ ]:


# converting the test data into numerical and then normalizing the data.
def encoding(dataframe_feature):
    if(dataframe_feature.dtype == 'object'):
        return LabelEncoder().fit_transform(dataframe_feature)
    else:
        return dataframe_feature
test_dataset = test_dataset.apply(encoding)


# In[ ]:


print(train_dataset.shape)
print(test_dataset.shape)


# In[ ]:


SalePrice = knn.predict(test_dataset)


# In[ ]:


submission_dataset = pd.DataFrame()
submission_dataset['Id'] = test_dataset.index
submission_dataset['SalePrice'] = SalePrice


# In[ ]:


submission_dataset.head()
# submission_dataset.to_csv("KNN_Housing_Regression.csv", index=False)


# I tried submitting it but it requires 1459 predicted rows. We dropped missing values. Now we will try imputing values on numerical dataset only and see the results.

# In[ ]:


# importing training and testing datasets again.
train_dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_dataset1 = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_dataset = test_dataset1.copy()


# In[ ]:


train_dataset = train_dataset.set_index(['Id'])
test_dataset = test_dataset.set_index(['Id'])


# In[ ]:


train_numerical_columns = train_dataset.describe().columns
train_dataset = train_dataset[train_numerical_columns]
train_dataset.head()


# In[ ]:


test_numerical_columns = test_dataset.describe().columns
test_dataset = test_dataset[test_numerical_columns]
test_dataset.head()


# In[ ]:


# importing the simple imputer missing values. 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')


# In[ ]:


imputed_data = imputer.fit_transform(train_dataset)
train_dataset = pd.DataFrame(data = imputed_data, columns = train_dataset.columns)
train_dataset.isnull().sum()


# In[ ]:


imputed_data = imputer.fit_transform(test_dataset)
test_dataset = pd.DataFrame(data = imputed_data, columns = test_dataset.columns)
test_dataset.isnull().sum()


# In[ ]:


X = train_dataset.drop(['SalePrice'], axis = 1)
y = train_dataset['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 22)


# In[ ]:


parameters = {
            'n_neighbors' : [1,2,3,4,5,6,7,8,9,10],
            'algorithm' : ['ball_tree', 'brute']
             }
grid_search_cv = GridSearchCV(KNeighborsRegressor(), parameters)
grid_search_cv.fit(X_train, y_train)
grid_search_cv.best_params_


# In[ ]:


knn = KNeighborsRegressor(n_neighbors = 6)
# fitting the model with the training dataset
knn.fit(X_train, y_train)
# predicting the values
predicted_value = knn.predict(X_test)
# calculating the accuracy
rmse = np.sqrt(mean_squared_error(predicted_value, y_test))
print(rmse)


# In[ ]:


print(train_dataset.shape)
print(test_dataset.shape)


# In[ ]:


SalePrice = knn.predict(test_dataset)


# In[ ]:


submission_dataset = pd.DataFrame({'Id' : test_dataset1['Id'], 'SalePrice': SalePrice})


# In[ ]:


submission_dataset.to_csv("KNN_Housing_Regression.csv", index=False)


# Being on Learning curve, I tried using basic KNN Regressor algorithm. Suggestions and Ideas on improving the model are appreciated.
