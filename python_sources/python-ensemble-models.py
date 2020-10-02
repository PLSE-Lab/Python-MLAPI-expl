#!/usr/bin/env python
# coding: utf-8

# # House Prices Prediction
# [Andreina Torres ](https://www.linkedin.com/in/andreina-torres-25341565/)
# 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# The goal of this project predicts the sales price (SalePrice variable) in a data set of houses base on a set of characteristics that describe the house. This model could bel be useful to assign prices and future estimation of new houses. The real state agencies and anyone looking  to buy a new house can have an idea.
# 
# To aim this goal we will be implementing a Gradient Boosting model, over a file that includes 80,000 different houses.

# The list of the Packages to used and reading the data process.

# In[ ]:


# packages to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from statistics import mode


#Reading the data sets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# ## 1- Data Base
# The goal is first understand the data structured to evaluate possible variable transformations, and required data cleaning.
# Most of the variables a categorical and the variable to predict is a continuous variable.
# 
# ### 1.1 Data  Structure

# In[ ]:


#head of fist 5 rows
df_train.head()


# ### 1.2 variables in the file
# The file has in total 1460 row and 81 variables.

# In[ ]:


#to  get the size of the file
df_train.shape
#all column names
df_train.columns
#list of variables with its statistics
df_train.describe().transpose()


# ## 2-Data cleaning

# ### 2.1 Missing values check and inputation
# 
# The missing values on the  following variables could mean they don't have that attribute so we will input "none"
# * PoolQC: This one has a high percentage of missings (99.5%). We will keep it as If this is missing, it means the house doesn't have a pool. 
# * MiscFeature: Miscellaneous feature not covered in other categories. Missing values probably mean that there are no special features.
# * Alley: Type of alley access. Probably no alley access if missing
# * Fence: Fence quality. Probably no fence if missing. 
# * FireplaceQu: Fireplace quality. Probably no fireplace if missing.
# * Garage variables: If missing houses do not have a garage.
# * Basement variables:  If missing houses do not Basement.
# * MasVnrType/MasVnrArea:  If missing houses do not have Masonry veneer.
# 
# Additional imputation
# * LotFrontage: Linear feet of street connected to the property. is a numeric variable so we will input the mean.
# * Electrical: low missing (7%) and the should have an Electrical system so will input the due to the mode since is categorical.

# In[ ]:


#list of variables with missing values
df_train.columns[df_train.isnull().any()]
#data set only missing variable values
df_train_missing=df_train[df_train.columns[df_train.isnull().any()]]

#Create a new function for missing values:
def num_missing(x):
    return sum(x.isnull())
#Applying per row:
#df_train['nmissing']=df_train_missing.apply(num_missing, axis=1) #axis=1 defines that function is to be applied on each row
#Applying per column:
df_train_missing.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
df_train_missing_table=round(df_train_missing.apply(num_missing, axis=0)/1460,4)*100 #axis=0 defines that function is to be applied on each column
df_train_missing_table =pd.DataFrame({'MissingPercentage':df_train_missing_table})
df_train_missing_table


# In[ ]:


#None inputation
var1=['Alley','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
df_train[var1]=df_train[var1].fillna('NA')

#Mean inputation
df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())

#Mode inputation
df_train['Electrical']=df_train['Electrical'].fillna(mode(df_train['Electrical']))


# Creating the same process for Missing values check and inputation in the test file.

# In[ ]:


#list of variables with missing values
df_test.columns[df_test.isnull().any()]
#data set only missing variable values
df_test_missing=df_test[df_test.columns[df_test.isnull().any()]]

#Applying per column:
df_test_missing.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

#None inputation
var1=['BsmtQual','Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
df_test[var1]=df_test[var1].fillna('NA')

#Mean inputation
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean())
df_test['BsmtFinSF1']=df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())
df_test['BsmtFinSF2']=df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mean())
df_test['BsmtUnfSF']=df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())
df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())
df_test['GarageCars']=df_test['GarageCars'].fillna(df_test['GarageCars'].mean())
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mean())
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mean())

#Mode inputation
df_test['MSZoning']=df_test['MSZoning'].fillna(mode('MSZoning'))
df_test['BsmtHalfBath']=df_test['BsmtHalfBath'].fillna(0)
df_test['KitchenQual']=df_test['KitchenQual'].fillna('TA')
df_test['Functional']=df_test['Functional'].fillna(mode('Functional'))
df_test['BsmtFullBath']=df_test['BsmtFullBath'].fillna(0)
df_test['SaleType']=df_test['SaleType'].fillna(mode('SaleType'))


# ### 2.1 Checking and removing outliers
# There are some outliers in the file so,  we keep only the ones that are within +6 to -6 standard deviations in the column 'Data'.

# In[ ]:


#check outlier
sns.boxplot(x=df_train['SalePrice']);


# In[ ]:


#Remove Outlier
df_train=df_train[np.abs(df_train.SalePrice-df_train.SalePrice.mean()) <= (6*df_train.SalePrice.std())]


# ### 2.3 Creating New variables 

# New variables base on  other variables:
# * TT_SF: Total SF of the house.
# * TT_bathrooms: Total number of bathrooms.
# * TT_rooms: Total number of rooms

# In[ ]:


df_test['TT_SF']=df_test['1stFlrSF']+df_test['2ndFlrSF']+df_test['TotalBsmtSF']+df_test['GarageArea']+df_test['GrLivArea']
df_test['TT_bathrooms']=df_test['FullBath']+df_test['HalfBath']+df_test['BsmtFullBath']+df_test['HalfBath']
df_test['TT_rooms']=df_test['TotRmsAbvGrd']+df_test['BsmtFullBath']+df_test['BsmtHalfBath']

df_train['TT_SF']=df_train['1stFlrSF']+df_train['2ndFlrSF']+df_train['TotalBsmtSF']+df_train['GarageArea']+df_train['GrLivArea']
df_train['TT_bathrooms']=df_train['FullBath']+df_train['HalfBath']+df_train['BsmtFullBath']+df_train['HalfBath']
df_train['TT_rooms']=df_train['TotRmsAbvGrd']+df_train['BsmtFullBath']+df_train['BsmtHalfBath']


# In[ ]:


#delete the ID variable in train data
#del df_train['Id']


# ### 2.4 Checking lineal model assumsions

# The SalePrice variable does not follow a normal distribution.

# In[ ]:


#histogram
sns.distplot(df_train['SalePrice']);


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# Applying log transformation to get a normalized variable.

# In[ ]:


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])


# In[ ]:


#histogram
sns.distplot(df_train['SalePrice']);


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# Same process for other continuous variables.

# In[ ]:


#List of numeric variables
numeric_vars = df_train.dtypes[df_train.dtypes != "object"].index

# Check the skew of all numerical features
skewed_vars = df_train[numeric_vars].skew().sort_values(ascending=False)
skewness = pd.DataFrame({'Skew':skewed_vars})

#keeping only the one with high skewness
skewness = skewness[abs(skewness['Skew']) > 0.7]
skewness

#Transformation 
skewed_vars = skewness.index
df_train[skewed_vars] = np.log(df_train[skewed_vars]+0.001)
df_test[skewed_vars] = np.log(df_test[skewed_vars]+0.001)


# Creating the Dummy variables to use in the model of categorical data.

# In[ ]:


#convert categorical variable into dummy
df_train=pd.get_dummies(df_train)
df_test=pd.get_dummies(df_test)

#Sort column to get the same order 
final_train, final_test = df_train.align(df_test,join='inner',axis=1)


# ## 3- Exploratory data Analysis
# 
# The highest correlation of the SalePrice of the houses with are with the following variables:
# 
# * OverallQual: Rates the overall material and finish of the house (0 to 10).
# * Years built:  Original construction date.
# * Years remoadd: Remodel date (same as construction date if no remodeling or additions).
# * TotalBsmtSF	: Total square feet of basement area.
# * 1stFlrSF: First Floor square feet.
# * GrLivArea:	 Above grade (ground) living area square feet
# * FullBath: Full bathrooms above grade.
# * TotRmsAbvGrd:	Total rooms above grade (does not include bathrooms).
# * GarageCars: Size of garage in car capacity.
# * GarageArea:	Size of garage in square feet.

# In[ ]:


#correlation
corr_df=df_train.corr()

#list of variables with high correaltion with Sale price
high_corr_df=corr_df[corr_df['SalePrice']>0.5]
high_corr_df=high_corr_df[list(corr_df[corr_df['SalePrice']>0.5].index)]

#graph
heatmap_df=high_corr_df
plt.subplots(figsize=(10,8))
sns.heatmap(heatmap_df,annot=True);


# In[ ]:


#Distibution and correation graph of the variables with highest correaltion
sns.pairplot(df_train[['SalePrice','OverallQual','GarageCars','GrLivArea', 'GarageArea']]);


# ## 4- Fit a GradientBoostingRegressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# GradientBoostingRegressor:
# define pipeline
GBR_model = make_pipeline(GradientBoostingRegressor())
# cross validation score
score = cross_val_score(GBR_model,final_train, df_train.SalePrice, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * score.mean()))


# In[ ]:


# fit and make predictions
GBR_model.fit(final_train,df_train.SalePrice)
predictions= GBR_model.predict(final_test)


#  ## 5- Fit Ensemble models

# These are the set of models that will be use to create the Ensemble model, in the Prosess were compared, checked and removed each of them and others. 
# 
# * LinearRegression
# * Ridge
# * RandomForestRegressor
# * GradientBoostingRegressor
# * Lasso

# In[ ]:


# Import libaries for  models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import Lasso

# Metrics for root mean squared error
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


# Initialize models
lr = LinearRegression()
rd = Ridge()
rf = RandomForestRegressor(
    n_estimators = 12,
    max_depth = 3,
    n_jobs = -1)
gb = GradientBoostingRegressor()
lasso=Lasso(alpha =0.0005, random_state=1)

#nn = MLPRegressor(
#    hidden_layer_sizes = (90, 90),
#    alpha = 2.75)


# In[ ]:


# Initialize Ensemble
model = StackingRegressor(
    regressors=[rf, gb, rd, lasso],
    meta_regressor=lr)

# Fit the model on our data
model.fit(final_train, df_train.SalePrice);


# The prediction using the ensemble model.

# In[ ]:


# Predict training data
y_pred = model.predict(final_train)
print(sqrt(mean_squared_error(df_train.SalePrice, y_pred)))


# In[ ]:


# Predict test data
y_pred = model.predict(final_test)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ###  Lasso model checking
# The following step is how check some in Mean Absolute Error  in one of the models.

# In[ ]:


# Fit the model on our data
lasso.fit(final_train, df_train.SalePrice)

score = cross_val_score(lasso,final_train, df_train.SalePrice, scoring= 'neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * score.mean()))


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## 6-File to Summit gradientBoosting

# In[ ]:


submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': np.exp(predictions)})
submission.to_csv('submissionGB.csv', index=False)
submission.head()


# ## 5-File to summit Ensemble models

# Final data for summition, since gradient boosting returns a higher error in the crossvalidation.

# In[ ]:


# Create empty submission dataframe
submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': np.exp(y_pred)})

# Output submission file
submission.to_csv('submissionEM.csv',index=False)
submission.head()


# 
