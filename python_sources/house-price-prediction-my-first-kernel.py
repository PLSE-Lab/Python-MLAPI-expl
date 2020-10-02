#!/usr/bin/env python
# coding: utf-8

#  Hello everyone, this is my first kernel. I will appreciate any kind of constructive feedback. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import cross_validation, metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# In[ ]:


#Read files:
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


print(train_df.columns.values)


# Which features are categorical? Which features are numerical?

# In[ ]:


# preview the data
train_df.head(10)


# Let's get a detailed distribution of numerical feature values across the samples.

# In[ ]:


train_df.describe() #Get summary of numerical variables


# Next, let's get a detailed distribution of categorical features across the samples.

# In[ ]:


train_df.describe(include=['O']) #Get summary of categorical variables


# Now, if you look at the data_description.txt, you can observe that a lot of categorical features contain **NA**  as a feature value. However, now they are considered as missing values.  For example the categorical feature **Alley** has **3** unique values: *Grvl*, *Pave* and *NA*. But in the above summary of cateforical features, **Alley** is seen to have **2** unique values as *NA* is considered as missing value. We will now rectify all those features.

# In[ ]:


# Here NA is a value for the feature Alley
train_df['Alley'].fillna('No_Alley',inplace=True)
test_df['Alley'].fillna('No_Alley',inplace=True)


# In[ ]:


# Here NA is a value for the feature BsmtQual
train_df['BsmtQual'].fillna('No_Basement',inplace=True)
test_df['BsmtQual'].fillna('No_Basement',inplace=True)


# In[ ]:


# Here NA is a value for the feature BsmtCond
train_df['BsmtCond'].fillna('No_Basement',inplace=True)
test_df['BsmtCond'].fillna('No_Basement',inplace=True)


# In[ ]:


# Here NA is a value for the feature BsmtExposure
train_df['BsmtExposure'].fillna('No_Basement',inplace=True)
test_df['BsmtExposure'].fillna('No_Basement',inplace=True)


# In[ ]:


# Here NA is a value for the feature BsmtFinType1
train_df['BsmtFinType1'].fillna('No_Basement',inplace=True)
test_df['BsmtFinType1'].fillna('No_Basement',inplace=True)


# In[ ]:


# Here NA is a value for the feature BsmtFinType2
train_df['BsmtFinType2'].fillna('No_Basement',inplace=True)
test_df['BsmtFinType2'].fillna('No_Basement',inplace=True)


# In[ ]:


# Here NA is a value for the feature FireplaceQu
train_df['FireplaceQu'].fillna('No_Fireplace',inplace=True)
test_df['FireplaceQu'].fillna('No_Fireplace',inplace=True)


# In[ ]:


# Here NA is a value for the feature GarageType
train_df['GarageType'].fillna('No_Garage',inplace=True)
test_df['GarageType'].fillna('No_Garage',inplace=True)


# In[ ]:


# Here NA is a value for the feature GarageFinish
train_df['GarageFinish'].fillna('No_Garage',inplace=True)
test_df['GarageFinish'].fillna('No_Garage',inplace=True)


# In[ ]:


# Here NA is a value for the feature GarageQual
train_df['GarageQual'].fillna('No_Garage',inplace=True)
test_df['GarageQual'].fillna('No_Garage',inplace=True)


# In[ ]:


# Here NA is a value for the feature GarageCond
train_df['GarageCond'].fillna('No_Garage',inplace=True)
test_df['GarageCond'].fillna('No_Garage',inplace=True)


# In[ ]:


# Here NA is a value for the feature PoolQC
train_df['PoolQC'].fillna('No_Pool',inplace=True)
test_df['PoolQC'].fillna('No_Pool',inplace=True)


# In[ ]:


# Here NA is a value for the feature Fence
train_df['Fence'].fillna('No_Fence',inplace=True)
test_df['Fence'].fillna('No_Fence',inplace=True)


# In[ ]:


# Here NA is a value for the feature MiscFeature
train_df['MiscFeature'].fillna('None',inplace=True)
test_df['MiscFeature'].fillna('None',inplace=True)


# In[ ]:


# Here None is a value for the feature MasVnrType
train_df['MasVnrType'].fillna('None',inplace=True)
test_df['MasVnrType'].fillna('None',inplace=True)


# Now, let's get a detailed distribution of categorical features across the samples. This time we will observe all such features has one more unique value than the previous one.

# In[ ]:


train_df.describe(include=['O']) #Get summary of categorical variables


# We will perform some visualization to better understand the distribution of different features and also the relationship of our dependent variable *SalePrice* with other features.

# In[ ]:


# plotting the histogram of GrLivArea
train_df['GrLivArea'].hist(bins=50)
plt.show()


# In[ ]:


# plotting the histogram of GarageArea
train_df['GarageArea'].hist(bins=50)
plt.show()


# In[ ]:


#scatter plot BedroomAbvGr/saleprice
var = 'BedroomAbvGr'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#scatter plot OverallQual/saleprice
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#scatter plot OverallCond/saleprice
var = 'OverallCond'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# Next, we will create some new features from the combination of old features and then drop the old features. For example, we can combine *OverallCond* and *OverallQual* to get a new feature as that will be more relevant.

# In[ ]:


#creating new_variable
train_df['Overallscore']=train_df['OverallQual']+train_df['OverallCond']
test_df['Overallscore']=test_df['OverallQual']+test_df['OverallCond']
#dropping irrevalent variables
train_df = train_df.drop(['OverallQual', 'OverallCond'], axis=1)
test_df = test_df.drop(['OverallQual', 'OverallCond'], axis=1)


# In[ ]:


#scatter plot Overallscore/saleprice
var = 'Overallscore'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='green');


# In[ ]:


#creating new_variable
train_df['BsmtFin']=train_df['BsmtFinSF1']+train_df['BsmtFinSF2']
test_df['BsmtFin']=test_df['BsmtFinSF1']+test_df['BsmtFinSF2']
#dropping irrevalent variables
train_df = train_df.drop(['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF'], axis=1)
test_df = test_df.drop(['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF'], axis=1)


# In[ ]:


# converting some categorical features to ordinal
Garage_mapping = {"No_Garage": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train_df['GarageQual'] = train_df['GarageQual'].map(Garage_mapping)
test_df['GarageQual'] = test_df['GarageQual'].map(Garage_mapping)
train_df['GarageCond'] = train_df['GarageCond'].map(Garage_mapping)
test_df['GarageCond'] = test_df['GarageCond'].map(Garage_mapping)
#creating new_variable
train_df['Garagescore']=train_df['GarageQual']+train_df['GarageCond']
test_df['Garagescore']=test_df['GarageQual']+test_df['GarageCond']
#dropping irrevalent variables
train_df = train_df.drop(['GarageQual', 'GarageCond'], axis=1)
test_df = test_df.drop(['GarageQual', 'GarageCond'], axis=1)
train_df['Garagescore'].head(10)


# In[ ]:


# converting some categorical features to ordinal
Bsmt_mapping = {"No_Basement": 0, "Unf":1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ":6}
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].map(Bsmt_mapping)
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].map(Bsmt_mapping)
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].map(Bsmt_mapping)
test_df['BsmtFinType2'] = test_df['BsmtFinType2'].map(Bsmt_mapping)
#creating new_variable
train_df['BsmtFinType']=train_df['BsmtFinType1']+train_df['BsmtFinType2']
test_df['BsmtFinType']=test_df['BsmtFinType1']+test_df['BsmtFinType2']
#dropping irrevalent variables
train_df = train_df.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1)
test_df = test_df.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1)
train_df['BsmtFinType'].head(10)


# In[ ]:


# converting some categorical features to ordinal
Exter_mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
train_df['ExterQual'] = train_df['ExterQual'].map(Exter_mapping)
test_df['ExterQual'] = test_df['ExterQual'].map(Exter_mapping)
train_df['ExterCond'] = train_df['ExterCond'].map(Exter_mapping)
test_df['ExterCond'] = test_df['ExterCond'].map(Exter_mapping)
#creating new_variable
train_df['Exterscore']=train_df['ExterQual']+train_df['ExterCond']
test_df['Exterscore']=test_df['ExterQual']+test_df['ExterCond']
#dropping irrevalent variables
train_df = train_df.drop(['ExterQual', 'ExterCond'], axis=1)
test_df = test_df.drop(['ExterQual', 'ExterCond'], axis=1)
train_df['Exterscore'].head(10)


# In[ ]:


#dropping irrevalent variables
train_df = train_df.drop(['Condition2'], axis=1)
test_df = test_df.drop(['Condition2'], axis=1)


# There are multiple features that have **year** as feature value. Instead of keeping the entire year as feature value, I have converted it to a number as 2018 - **year**

# In[ ]:


# Determining the years from Original construction date
# Years:
train_df['Age_of_House'] = 2018 - train_df['YearBuilt']
test_df['Age_of_House'] = 2018 - test_df['YearBuilt']
train_df['Age_of_House'].describe()


# In[ ]:


#scatter plot Age_of_House/saleprice
var = 'Age_of_House'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='green');


# In[ ]:


# dropping YearBuilt feature
train_df = train_df.drop(['YearBuilt'], axis=1)
test_df = test_df.drop(['YearBuilt'], axis=1)


# In[ ]:


# Determining the years from Remodelling date
# Years:
train_df['Age_of_Remod_House'] = 2018 - train_df['YearRemodAdd']
test_df['Age_of_Remod_House'] = 2018 - test_df['YearRemodAdd']
train_df['Age_of_Remod_House'].describe()


# In[ ]:


#scatter plot Age_of_Remod_House/saleprice
var = 'Age_of_Remod_House'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='green');


# In[ ]:


# dropping YearRemodAdd feature
train_df = train_df.drop(['YearRemodAdd'], axis=1)
test_df = test_df.drop(['YearRemodAdd'], axis=1)


# In[ ]:


# Determining the years from garage built date
# Years:
train_df['Age_of_garage'] = 2018 - train_df['GarageYrBlt']
test_df['Age_of_garage'] = 2018 - test_df['GarageYrBlt']
train_df['Age_of_garage'].describe()


# In[ ]:


#scatter plot Age_of_House/saleprice
var = 'Age_of_garage'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='green');


# In[ ]:


# dropping GarageYrBlt feature
train_df = train_df.drop(['GarageYrBlt'], axis=1)
test_df = test_df.drop(['GarageYrBlt'], axis=1)


# Dropping some more irrevalent features

# In[ ]:


#scatter plot YrSold/saleprice
var = 'YrSold'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#dropping irrevalent variables
train_df = train_df.drop(['YrSold', 'MoSold'], axis=1)
test_df = test_df.drop(['YrSold', 'MoSold'], axis=1)


# In[ ]:


#scatter plot LotFrontage/saleprice
var = 'LotFrontage'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#dropping irrevalent variables
train_df = train_df.drop(['LotFrontage'], axis=1)
test_df = test_df.drop(['LotFrontage'], axis=1)


# In[ ]:


#scatter plot MasVnrArea/saleprice
var = 'MasVnrArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#dropping irrevalent variables
train_df = train_df.drop(['MasVnrArea'], axis=1)
test_df = test_df.drop(['MasVnrArea'], axis=1)


# In[ ]:


#scatter plot GarageCars/saleprice
var = 'GarageCars'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#scatter plot GarageArea/saleprice
var = 'GarageArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), color='red');


# In[ ]:


#dropping irrevalent variables
train_df = train_df.drop(['GarageArea'], axis=1)
test_df = test_df.drop(['GarageArea'], axis=1)


# **Missing data**
# Let's analyze the missing data for training set and test set

# In[ ]:


#missing data for training set
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[ ]:


#missing data for test set
total = test_df.isnull().sum().sort_values(ascending=False)
percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(15)


# **Age_of_garage** is missing where there is no garage so we can fill the missing values by 0.  Same is the case for **BsmtFin**, **BsmtUnfSF** and **GarageCars**. For rest of the features, we will simply fill the missing values by mode value of that feature.

# In[ ]:


# Age_of_garage is missing where there is no garage
train_df['Age_of_garage'].fillna(0,inplace=True)
test_df['Age_of_garage'].fillna(0,inplace=True)


# In[ ]:


#Treating missing values
train_df['Electrical'].fillna(train_df['Electrical'].mode()[0],inplace=True)
test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0],inplace=True)
test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0],inplace=True)
test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0],inplace=True)
test_df['Functional'].fillna(test_df['Functional'].mode()[0],inplace=True)
test_df['Utilities'].fillna(test_df['Utilities'].mode()[0],inplace=True)
test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0],inplace=True)
test_df['SaleType'].fillna(test_df['SaleType'].mode()[0],inplace=True)
test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0],inplace=True)
test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0],inplace=True)
test_df['BsmtFin'].fillna(0,inplace=True)
test_df['BsmtUnfSF'].fillna(0,inplace=True)
test_df['GarageCars'].fillna(0,inplace=True)


# Next, we convert all categorical variables into numeric by Label Encoding

# In[ ]:


# Converting all categorical variables into numeric by encoding the categories
var_mod = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','BsmtQual','BsmtCond','BsmtExposure','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
le = LabelEncoder()
for i in var_mod:
    train_df[i] = le.fit_transform(train_df[i])
    test_df[i] = le.fit_transform(test_df[i])
train_df.dtypes


# In[ ]:


#dropping irrevalent variables
train_df = train_df.drop(['Id'], axis=1)


# In[ ]:


X_train = train_df.drop("SalePrice", axis=1)
Y_train = train_df["SalePrice"]
X_test  = test_df.drop("Id", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# As you can see, we have 67 features currently which is huge. We will further reduce the dimensionality using *Principal Component Analysis(PCA).  *

# In[ ]:


pca = PCA(n_components=30)
X_train_reduced=pca.fit_transform(X_train)
X_train_reduced.shape


# In[ ]:


X_test_reduced=pca.fit_transform(X_test)
X_test_reduced.shape


# We will define a generic function for model fitting and then we will fit models like *Linear Regression, Ridge Regression, Decision Tree Regressor, Random Forest Regressor *and *XGB regressor *

# In[ ]:


# generic function
def modelfit(alg, dtrain_X, dtrain_Y, dtest_X):
    #Fit the algorithm on the data
    alg.fit(dtrain_X, dtrain_Y)
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain_X)

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain_X, dtrain_Y, cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain_Y.values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    Y_pred = alg.predict(dtest_X)
    
    return Y_pred


# In[ ]:


# Linear Regression Model
alg1 = LinearRegression(normalize=True)
Y_pred=modelfit(alg1, X_train_reduced, Y_train, X_test_reduced)


# In[ ]:


# Ridge Regression Model
alg2 = Ridge(alpha=0.05,normalize=True)
Y_pred=modelfit(alg2, X_train_reduced, Y_train, X_test_reduced)


# In[ ]:


# Decision Tree Model
alg3 = DecisionTreeRegressor(max_depth=20, min_samples_leaf=300)
Y_pred=modelfit(alg3, X_train_reduced, Y_train, X_test_reduced)


# In[ ]:


# Random Forest Model
alg4 = RandomForestRegressor(n_estimators=400,max_depth=20, min_samples_leaf=100, n_jobs=4)
Y_pred=modelfit(alg4, X_train_reduced, Y_train, X_test_reduced)


# In[ ]:


# XGB regressor
alg5 = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.1) 
Y_pred=modelfit(alg5, X_train_reduced, Y_train, X_test_reduced)


# XGB regressor outperforms other models. So, we will tune its parameters to further improve the results.

# In[ ]:


# Tuning of parameters
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.3],
    'n_estimators': [200, 350, 450, 500]
}
# Create a based model
XGBR = xgb.XGBRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = XGBR, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train_reduced, Y_train)
grid_search.best_params_


# In[ ]:


# XGB regressor with tuned parameters
alg6 = xgb.XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.05) 
Y_pred=modelfit(alg6, X_train_reduced, Y_train, X_test_reduced)


# Next we will use Artificial Neural Network with two hidden layers.

# In[ ]:


# Using ANN
# Initialising the ANN
alg7 = Sequential()

# Adding the input layer and the first hidden layer
alg7.add(Dense(units = 256, kernel_initializer = 'normal', activation = 'relu', input_dim = 30))

# Adding the second hidden layer
alg7.add(Dense(units = 256, kernel_initializer = 'normal', activation = 'relu'))

# Adding the output layer
alg7.add(Dense(units = 1, kernel_initializer = 'normal'))

# Compiling the ANN
alg7.compile(optimizer = 'adam', loss = 'mse')

# Fitting the ANN to the Training set
alg7.fit(X_train_reduced, Y_train, batch_size = 64, epochs = 800)

# Predicting the Test set results
y_pred = alg7.predict(X_test_reduced)


# But the neural network has not outperformed our XGB regressor. So, we will submit the predictions made by XGB regressor.

# In[ ]:


submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": Y_pred
    })

submission.to_csv('submission.csv', index=False)

