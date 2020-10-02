#!/usr/bin/env python
# coding: utf-8

# # Dealing with missing values

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Reading the data
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


#Analyzing the data
train_data


# To ensure that the same pre-processing will be done in the training and testing sets, we will join them and separate later.
# 
# So, the first pre-processing step will be removing the Id labels because it is just the identifier of the house, so it's not a feature.

# In[ ]:


#Insert the Id Column with -1 value
test_data = test_data.set_index(test_data['Id']-1)
test_data.insert(test_data.shape[1], 'SalePrice', -1)
pre_processing_data = pd.concat([train_data, test_data], axis=0)

#Removing the id layer.
pre_processing_data = pre_processing_data.drop(columns=['Id'])
test_data


# In[ ]:


pre_processing_data.info()


# Here we can see that there are some features that have some missing values. We will separate them into two groups, those which the missing values represents the absence of the feature (like a zero value)(atributes_NAN_normal), and the others which the missing value represents a problem in the collection of the data.

# In[ ]:


#Atributes_NAN contains all the features that has NAN values. NAN values do not implies missing values, so
#after collecting the NAN features, we are going to select just the features that have missing values.
atributes_NAN = pre_processing_data.iloc[:, (pre_processing_data.isna().sum() > 0).to_numpy()].columns
atributes_NAN_normal = np.array(['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                                 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                                 'GarageCond', 'PoolQC','Fence', 'MiscFeature'])
atributes_NAN_missing_values = atributes_NAN.drop(atributes_NAN_normal)

print("Total number of features with missing values: " + str(atributes_NAN.shape))
print("Total number of features that have normal missing values: " + str(atributes_NAN_normal.shape))
print("Total number of features that have real missing values: " + str(atributes_NAN_missing_values.shape))
print(atributes_NAN_missing_values)


# With these two sets of missing values, we will plot the number of missing values of every feature.

# In[ ]:


x = pre_processing_data.columns
y = pre_processing_data.isna().sum().values

plt.figure(figsize=(17, 5))
plt.xticks(
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='medium'  
)

plt.bar(x, y)


# There are 4 features that have a lot of missing values. This is not necessarilly real missing values, but even in the normal missing values, we will have features that have low variance which is not a desired characteristic. Therefore let's drop them.

# In[ ]:


missing_values_dropped = pre_processing_data.iloc[:, (pre_processing_data.isna().sum() > 
                                                     pre_processing_data.shape[0]/2).to_numpy()].columns
pre_processing_data = pre_processing_data.drop(columns=missing_values_dropped)

print("New shape: " + str(pre_processing_data.shape))

#New features
atributes_NAN = atributes_NAN.drop(missing_values_dropped)
atributes_NAN_normal = np.array(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                                 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                                 'GarageCond'])


# Now, let's replace some missing values by a default. Initially we will change the value of the normal missing values by zero.

# In[ ]:


#Replacing the NAN values in the features which the NAN represent "no feature", by zero
print(pre_processing_data.shape)
pre_processing_data[atributes_NAN_normal] = pre_processing_data[atributes_NAN_normal].fillna(0)
pre_processing_data.info()


# Now we have to deal with the real missing values. So for a better analysis, let's print the histograms of those five features.

# In[ ]:


#Plotting the histograms for the numerical data
print(pre_processing_data[atributes_NAN_missing_values].info())
pre_processing_data[atributes_NAN_missing_values].hist()
plt.tight_layout()


# In[ ]:


MSZoning_values = np.delete(pre_processing_data['MSZoning'].unique(), 
                              len(pre_processing_data['MSZoning'].unique())-1)
Utilities_values = np.delete(pre_processing_data['Utilities'].unique(), 
                              len(pre_processing_data['Utilities'].unique())-1)
Exterior1st_values = np.delete(pre_processing_data['Exterior1st'].unique(), 
                              len(pre_processing_data['Exterior1st'].unique())-1)
Exterior2nd_values = np.delete(pre_processing_data['Exterior2nd'].unique(), 
                              len(pre_processing_data['Exterior2nd'].unique())-1)
MasVnrType_values = np.delete(pre_processing_data['MasVnrType'].unique(), 
                              len(pre_processing_data['MasVnrType'].unique())-1)
Electrical_values = np.delete(pre_processing_data['Electrical'].unique(), 
                              len(pre_processing_data['Electrical'].unique())-1)
KitchenQual_values = np.delete(pre_processing_data['KitchenQual'].unique(), 
                              len(pre_processing_data['KitchenQual'].unique())-1)
Functional_values = np.delete(pre_processing_data['Functional'].unique(), 
                              len(pre_processing_data['Functional'].unique())-1)
SaleType_values = np.delete(pre_processing_data['SaleType'].unique(), 
                              len(pre_processing_data['SaleType'].unique())-1)

fig, ax = plt.subplots(3, 3)

MasVnrType_histogram = pre_processing_data.MasVnrType.value_counts()
Electrical_histogram = pre_processing_data.Electrical.value_counts()
MSZoning_histogram = pre_processing_data.MSZoning.value_counts()
Utilities_histogram = pre_processing_data.Utilities.value_counts()
Exterior1st_histogram = pre_processing_data.Exterior1st.value_counts()
Exterior2nd_histogram = pre_processing_data.Exterior2nd.value_counts()
KitchenQual_histogram = pre_processing_data.KitchenQual.value_counts()
Functional_histogram = pre_processing_data.Functional.value_counts()
SaleType_histogram = pre_processing_data.SaleType.value_counts()

ax[0, 0].set_title('MasVnrType')
ax[0, 0].bar(MasVnrType_values, MasVnrType_histogram)
ax[0, 1].set_title('Electrical')
ax[0, 1].bar(Electrical_values, Electrical_histogram)
ax[0, 2].set_title('MSZoning')
ax[0, 2].bar(MSZoning_values, MSZoning_histogram)
ax[1, 0].set_title('Utilities')
ax[1, 0].bar(Utilities_values, Utilities_histogram)
ax[1, 1].set_title('Exterior1st')
ax[1, 1].bar(Exterior1st_values, Exterior1st_histogram)
ax[1, 2].set_title('Exterior2nd')
ax[1, 2].bar(Exterior2nd_values, Exterior2nd_histogram)
ax[2, 0].set_title('KitchenQual')
ax[2, 0].bar(KitchenQual_values, KitchenQual_histogram)
ax[2, 1].set_title('Functional')
ax[2, 1].bar(Functional_values, Functional_histogram)
ax[2, 2].set_title('SaleType')
ax[2, 2].bar(SaleType_values, SaleType_histogram)

plt.tight_layout()
plt.xticks(
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='medium'  
)


# We won't dealing the with all these features in the same way. Here is a list of the approach:
# 
# 1. MarsVnrType: replace the nan with BrkFace because is the most frequent and it has just 8 missing values.
# 2. Electrical: replace the nan with SBrkr because of the same argument.
# 3. LotFrontage: there are 259 missing values, so we have to use another strategy. Replace with the median value.  
# 4. MasVnrArea: there are 8 missing values. Replace with the most frequent value.
# 5. GarageYrBlt: there are 81 missing values. Replace with the 1978 which is the median value.
# 6. BsmtFinSF1: There are 1 missing values. Replace with the median value.
# 7. BsmtFinSF2: There are 1 missing values. Replace with the median value.
# 8. BsmtFullBath: There are 2 missing values. Replace with the median value.
# 9. BsmtHalfBath: There are 2 missing values. Replace with the median value.
# 10. BsmtUnfSF: There are 1 missing values. Replace with the median value.
# 11. GarageArea: There are 1 missing values. Replace with the median value.
# 12. GarageCars: There are 1 missing values. Replace with the median value.
# 13. TotalBsmtSF: There are 1 missing values. Replace with the median value.
# 14. MSZoning: There are 4 missing values. Replace with the 'RL'.
# 15. Utilities: There are 4 missing values. Replace with the 'AllPub'. 
# 16. Exterior1st: There are 1 missing values. Replace with the 'VinylSd'.
# 17. Exterior2nd: There are 1 missing values. Replace with the 'VinylSd'.
# 18. KitchenQual: There are 1 missing values. Replace with the 'Gd'.
# 19. Functional: There are 2 missing values. Replace with the 'Typ'.
# 20. SaleType: There are 2 missing values. Replace with the 'WD'.

# In[ ]:


pre_processing_data['MasVnrType'] = pre_processing_data['MasVnrType'].fillna('BrkFace')
pre_processing_data['Electrical'] = pre_processing_data['Electrical'].fillna('Electrical')
pre_processing_data['LotFrontage'] = pre_processing_data['LotFrontage'].fillna(int(
                                     np.median(pre_processing_data['LotFrontage'].dropna())))
pre_processing_data['MasVnrArea'] = pre_processing_data['MasVnrArea'].fillna(stats.mode(
                                     pre_processing_data['MasVnrArea']).mode[0])
pre_processing_data['GarageYrBlt'] = pre_processing_data['GarageYrBlt'].fillna(int(np.mean(
                                     pre_processing_data['GarageYrBlt'].dropna())))
pre_processing_data['BsmtFinSF1'] = pre_processing_data['BsmtFinSF1'].fillna(int(np.median(
                                     pre_processing_data['BsmtFinSF1'].dropna())))
pre_processing_data['BsmtFinSF2'] = pre_processing_data['BsmtFinSF2'].fillna(int(np.median(
                                     pre_processing_data['BsmtFinSF2'].dropna())))
pre_processing_data['BsmtFullBath'] = pre_processing_data['BsmtFullBath'].fillna(int(np.median(
                                     pre_processing_data['BsmtFullBath'].dropna())))
pre_processing_data['BsmtHalfBath'] = pre_processing_data['BsmtHalfBath'].fillna(int(np.median(
                                     pre_processing_data['BsmtHalfBath'].dropna())))
pre_processing_data['BsmtUnfSF'] = pre_processing_data['BsmtUnfSF'].fillna(int(np.median(
                                     pre_processing_data['BsmtUnfSF'].dropna())))
pre_processing_data['GarageArea'] = pre_processing_data['GarageArea'].fillna(int(np.median(
                                     pre_processing_data['GarageArea'].dropna())))
pre_processing_data['GarageCars'] = pre_processing_data['GarageCars'].fillna(int(np.median(
                                     pre_processing_data['GarageCars'].dropna())))
pre_processing_data['TotalBsmtSF'] = pre_processing_data['TotalBsmtSF'].fillna(int(np.median(
                                     pre_processing_data['TotalBsmtSF'].dropna())))
pre_processing_data['MSZoning'] = pre_processing_data['MSZoning'].fillna('RL')
pre_processing_data['Utilities'] = pre_processing_data['Utilities'].fillna('AllPub')
pre_processing_data['Exterior1st'] = pre_processing_data['Exterior1st'].fillna('VinylSd')
pre_processing_data['Exterior2nd'] = pre_processing_data['Exterior2nd'].fillna('VinylSd')
pre_processing_data['KitchenQual'] = pre_processing_data['KitchenQual'].fillna('Gd')
pre_processing_data['Functional'] = pre_processing_data['Functional'].fillna('Typ')
pre_processing_data['SaleType'] = pre_processing_data['SaleType'].fillna('WD')


# In[ ]:


correlation = pre_processing_data.corr()
correlation


# Now let's eliminate those features that have low correlation with the SalePrice.

# In[ ]:


#uncorrelated_columns = (np.abs(correlation['SalePrice']) < 0.25).index
#train_data = train_data[uncorrelated_columns]

#uncorrelated_columns


# In[ ]:


pre_processing_data.info()


# Now, we don't have missing values. So we have to convert the categorical data into a numerical for inserting the features in the machine learning algorithm.

# # Convert categorical data into numerical

# First we have to count how many features are categorical and how many of them have a sense of order (Ordinal Features).

# In[ ]:


attributes_types_categorical = {}
for c in pre_processing_data.columns:
    if pre_processing_data[c].values.dtype == 'O':
        attributes_types_categorical[c] = pre_processing_data[c].values.dtype 
    
print("Number of Categorical Features: " + str(len(attributes_types_categorical)))
print("Number of Numerical Features: " + str(len(pre_processing_data.columns) - 
                                             len(attributes_types_categorical)))


# In[ ]:


categorical_features = np.array(list(attributes_types_categorical.keys()), dtype=object)

nominal_features = np.array(['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                             'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                             'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                             'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 
                             'SaleType', 'SaleCondition'])

indexes = np.in1d(categorical_features, nominal_features).nonzero()[0]
ordinal_features = np.delete(categorical_features, indexes)

print("Number of Nominal Features: " + str(len(nominal_features)))
print(nominal_features)
print()
print("Number of Ordinal Features: " + str(len(ordinal_features)))
print(ordinal_features)


# With these numbers we have to encode the ordinal variables case by case, and for the nominal variables we are going to use the one-hot encoding.

# In[ ]:


def encoding_ordinal_features(ordinal_features_names, data):
    ordinal_encoder = OrdinalEncoder()
    
    new_data = data.copy()
    new_data[ordinal_features_names] = ordinal_encoder.fit_transform(data[ordinal_features_names].astype(str))
    return new_data

def encoding_nominal_features(nominal_features_names, data):
    one_hot_encoder = OneHotEncoder()
    
    one_hot_nominal_arr = one_hot_encoder.fit_transform(data[nominal_features_names].astype(str)).toarray()
    nominal_feature_labels = list(one_hot_encoder.categories_)
    
    flat_list = []
    for sublist in nominal_feature_labels:
        for item in sublist:
            flat_list.append(item)
    
    #The Electrical feature has an Electrical value, so for avoiding the conflict, we will change the name
    flat_list[flat_list.index('Electrical')] = 'Electrical0'
    nominal_features_pd = pd.DataFrame(one_hot_nominal_arr, columns=flat_list)
    new_data = pd.concat([data, nominal_features_pd], axis=1)
    return new_data

pre_processing_data = encoding_ordinal_features(ordinal_features, pre_processing_data)
pre_processing_data = encoding_nominal_features(nominal_features, pre_processing_data)

#Now we eliminate the original nominal features
pre_processing_data = pre_processing_data.drop(columns=nominal_features)
pre_processing_data.info()


# Let's normalize the data

# In[ ]:


scaler= RobustScaler()

Y_train_original = pre_processing_data['SalePrice']
Y_train_original = Y_train_original.iloc[0:1460]
pre_processing_original = pre_processing_data.drop(columns='SalePrice')

pre_processing_original = scaler.fit_transform(pre_processing_original)


# Now, let's separate some data for the validation set

# In[ ]:


X_train_original = pre_processing_original[0:1460, 0:pre_processing_original.shape[1]]
X_test_original = pre_processing_original[1460:, 0:pre_processing_original.shape[1]]

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_original, 
                                                                Y_train_original, 
                                                                random_state=42, 
                                                                test_size=0.30)
print(X_train.shape)
print(Y_train.shape)


# Finally let's train some models

# In[ ]:


lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train, Y_train)

Y_predicted = lin_reg.predict(X_validation)
mse_lin_reg = np.sqrt(mean_squared_log_error(Y_validation, Y_predicted))
print("Linear Regression")
print("MSE Linear Regression: " + str(mse_lin_reg))


# In[ ]:


lin_ridge = linear_model.Ridge(alpha=0.5)
lin_ridge.fit(X_train, Y_train)

Y_predicted = lin_ridge.predict(X_validation)
mse_lin_ridge = np.sqrt(mean_squared_log_error(Y_validation, Y_predicted))
print("Ridge")
print("MSE Ridge Regression: " + str(mse_lin_ridge))


# In[ ]:


lin_lasso = linear_model.Lasso(alpha=33)
lin_lasso.fit(X_train, Y_train)

Y_predicted = lin_lasso.predict(X_validation)
mse_lin_lasso = np.sqrt(mean_squared_log_error(Y_validation, Y_predicted))
print("Lasso")
print("MSE Lasso Regression: " + str(mse_lin_lasso))


# In[ ]:


RFR = RandomForestRegressor(max_depth=50)
RFR.fit(X_train, Y_train)

Y_predicted = RFR.predict(X_validation)
mse_random_forest = np.sqrt(mean_squared_log_error(Y_validation, Y_predicted))
print("Random Forest")
print("MSE Random Forest: " + str(mse_random_forest))


# # Evaluating the Best Model in the Test Data

# Since the Lasso model with alpha equals to 33 was the best model, we will evaluate that in the test data and submit to the competition.

# In[ ]:


Y_test_predicted = RFR.predict(X_test_original)
indexes = np.arange(X_test_original.shape[0]+2, 2*X_test_original.shape[0]+2)
submission = pd.DataFrame({'Id': indexes, 'SalePrice': Y_test_predicted})
submission.to_csv('submission.csv', index=False)
print(submission)

