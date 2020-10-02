#!/usr/bin/env python
# coding: utf-8

# ***Introduction***

# Hi guys, my name is Ivan. 
# This is my very first machine learning kernel and I've achieved a score of 0.12309. 
# In this kernel, I will mainly discuss on the fundamental concepts of machine learning with minimal codes. As I'm not a programmer, please let me know if there is a better alternative/approach. Appreciate with that!
# 
# **Here is the main content of this kernel:**
# * Import packages and training & testing set 
# * Data Cleaning Process  
#       1. Remove outliers
#       2. Fille up missing values
#       3. Drop redundant variables
#       4. Categorize variables
#       5. Deal with interaction between the variables
# * Find the best fit model
#       1. Lasso Regression
#       2. Ridge Regression
# * Export the result

# ***Part I: Import packages and training & testing set ***

# In[ ]:


#Import basic packages
import pandas as pd
import numpy as np
#Import packages for preprocessing (data cleaning)
from sklearn.preprocessing import RobustScaler, LabelEncoder
#Import packages for data visualisation
import matplotlib.pyplot as plt
#Import packages for model testing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#Import packages for modeling
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline


# In[ ]:


#Import both training & testing set into Python. 
df_train_ori = pd.read_csv('../input/train.csv')
df_test_ori = pd.read_csv('../input/test.csv')

#Print the shape of both training & testing set. It makes sense as the target variable (SalePrice) is only available in training set.
print(df_train_ori.shape)
print(df_test_ori.shape)


# After checking the shape of the dataset, I'm going to assign a new column of 'SalePrice' with value of 0 in testing set. In this case, when we combine the training & testing set together, it can be distinguished by the SalePrice. 

# In[ ]:


df_test_ori['SalePrice'] = 0


# In[ ]:


#In this project, as we are predicting the SalePrice of a property, it is a common sense that there should be a linear relationsip
#between the area of the property and the sale price of the property. (The area of the property is in cloumn 'GrLivArea')

#We will use scatter plot to see if the relationship is linear. 
plt.scatter(x = df_train_ori['GrLivArea'], y = df_train_ori['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# From scatter plot, the result is as expected. The SalePrice increases when the area of the property increases. We can also conclude that two points in the bottom right are definitely outliers. We can then remove the two points in the bottom right.

# In[ ]:


df_train_ori = df_train_ori.drop(df_train_ori[(df_train_ori['GrLivArea']>4000) & (df_train_ori['SalePrice']<300000)].index)


# In[ ]:


#Double check the shape of training set, in case we drop more points than expected. 
print(df_train_ori.shape)
#The result is as expected, only two points in the bottom right are dropped. 


# In[ ]:


#Combine the training&testing set
df_all = df_train_ori.append(df_test_ori)
#Double check the shape with the combined dataset. 
print(df_all.shape)


# **Part II: Data Cleaning Process**
# 
# 

# In[ ]:


#Check the info for the combined dataset. 
print(df_all.info())


# Missing Values
# 
# In this project, we can see that there are some NaN in many of the columns. Basically there are mainly two approaches to deal with NaN. 
# 1. Drop all the columns with NaN and only use those columns without NaN for further analysis. 
# 2. Use 'fillna' method to fill up the missing values in a reasonbly manner (use your own judgement). 
# 
# In this case, we will use the second approach. As if we use the first approach, many of the columns will be dropped and the result is expected to be less accurate. Once again, since I'm not proficient in Python,  I will have to a lot of manual work from now on. The data description provided by this project is very useful. 
# 
# Based on the data description, I manage to categorize all the features (with NaNs) as below; 
# 
# 1. Replace 'NaN' with 'None'
# 2. Replace 'NaN' with 0
# 3. Replace 'NaN' with the most frequent number/category of its corresponding feature. 
# 4. Replace 'NaN' with the mean/median of its corresponding feature.
# 
# The result of analysis is subjective to individuals, you may have a even better result than mine if the categorize it differently. Nevertheless, let's continue. 

# In[ ]:


#By looking at the training & testing set, the missing value in these columns should be filled with 'None'. 
for i in ['Alley', 'MasVnrType','BsmtQual','BsmtCond','FireplaceQu', 'GarageType','GarageFinish','GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    df_all[i] = df_all[i].fillna('None')


# In[ ]:


#The missing value in these columns should be filled with 0.
for i in ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','GarageYrBlt', 'GarageCars','GarageArea']:
    df_all[i] = df_all[i].fillna(0)


# In[ ]:


#The missing value in these columns should be filled with the mean of its corresponding feature.
df_all["LotFrontage"] = df_all["LotFrontage"].fillna(df_all["LotFrontage"].mean())


# In[ ]:


#The missing value in these columns should be filled with the most frequent number of its corresponding feature.
for i in ['MSZoning', 'Exterior1st','Exterior2nd','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'Electrical','BsmtFullBath', 'BsmtHalfBath','KitchenQual', 'Functional','SaleType']:
    df_all[i] = df_all[i].fillna(df_all[i].mode()[0])


# In[ ]:


#The missing value in these columns should be filled with the most frequent category of its corresponding feature.
df_all["Functional"] = df_all["Functional"].fillna("Typ")


# Drop redundant columns
# 
# Now we only left column 'Utilities'. 
# Below are some statistics. 
# Original Training set: one entry for 'NoSeWa' and the rest are 'AllPub'
# Original Testing set: 2 entries for 'NaNs' and the rest are 'AllPub' 
# 
# Given that the entry 'NoSeWa' is not available in testing set and there is no missing value in training set, this column is therefore redundant in forecasting the SalePrice of the property. Hence, we can drop the cloumn 'Utilities'. 
# 

# In[ ]:


df_all = df_all.drop(['Utilities'], axis = 1)


# Interation between the columns
# 
# After some further exploration on the dataset & data exploration, the interation between column 'TotalBsmtSF', '1stFlrSF' and '2ndFlrSF' should be taken into consideration.  Create a new column of 'TotalAreaSF' and its value will be the sum of TotalBsmtSF + 1stFlrSF + 2ndFlrSF. 

# In[ ]:


df_all['TotalAreaSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']


# Final Check on the shape of the combined dataset 
# 
# There is no longer NaNs in the combined dataset!

# In[ ]:


print(df_all.info())


# Categorize all the variables into different groups
# 
# In this dataset, as we are dealing with both numeric & categorical variables. We will have different approaches for different types of variables. After further exploration on the dataset, I manage to categorize all the variables into 3 different groups. 
# Group 1. Numeric Variables
# 2. Categorical Variables with equally ranking 
# 3. Categorical Variables with priority ranking
# 
# e.g.
# column 'RoofStyle', it is a categorical variable which represents different types of the roof of the property. In this case, each of the category should be treated equally. 
# column 'ExterQual', it is a categorical variable which evaluates the quality of the material on the exterior. The categories are: 'Excellent', 'Good', 'Average/Typical', 'Fair', 'Poor'; Each of the category should be treated differently as it contains information in terms of ordering. 
# 
# 
# Approaches:
# 1. Numeric Variables: Normalize all the numeric variables (both discrete and continuous) such that all the numeric variables would have the same weightage. 
# 2. Categorical Variables with equivalent ranking: Assign dummy variables (0/1) to represent each of the category in each of the variables. 
# 3. Categorical Variables with priority ranking: Assign different values to represent each of the category in each of the variables. Use encode() method. 
# 
# 

# Encoding categorical variables with priority ranking

# In[ ]:


#These variables are classified as categorical Variables with priority ranking
encode = ['LotShape','LandSlope','Neighborhood', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
          'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 
          'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
#Use encode() method to assign different numeric values to each of the category in each of the variable, 
for i in encode:
    le = LabelEncoder() 
    le.fit(list(df_all[i].values)) 
    df_all[i] = le.transform(list(df_all[i].values))


# Normalize numeric variables

# In[ ]:


#These variables are classified as numeric variables
norm = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond','YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'TotalAreaSF']

#Combine encode & norm to normalize all the numeric variables
norm_variables = norm + encode
df_all[norm_variables] = (df_all[norm_variables]-df_all[norm_variables].mean())/(df_all[norm_variables].max()-df_all[norm_variables].min())


# Assign dummy variable to categorical variables with equally ranking

# In[ ]:


#These variables are classified as categorical Variables with equivalent ranking
cat = ['MSZoning', 'Street', 'Alley','LandContour', 'LotConfig','Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature','SaleType',
       'SaleCondition']

df_all_dummy = pd.get_dummies(df_all, drop_first = True)
#This is the end of data cleaning process


# ** Part III: Find the best fit model**
# 
# Now that we have completed the data cleaning process, we can proceed to the next step: fittng the model. 
# Before that, we should split the combined dataset with training set & testing set. 

# In[ ]:


df_train_adj = df_all_dummy[df_all_dummy['SalePrice'] != 0]
df_test_adj = df_all_dummy[df_all_dummy['SalePrice'] == 0]


# In[ ]:


#Training the data
data_to_train = df_train_adj.drop(['SalePrice','Id'], axis = 1)


# Log the Target variable (SalePrice)
# 
# Info from evaluation: Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.

# In[ ]:


df_train_adj["SalePrice"] = np.log1p(df_train_adj["SalePrice"])
labels_to_use = df_train_adj['SalePrice']


# In[ ]:


#Build and fit the model
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))
ridge = make_pipeline(RobustScaler(), Ridge(alpha =20, random_state=42))


# **Evaluation on performance**

# In[ ]:


def evaluation(model):
    result= np.sqrt(-cross_val_score(model, data_to_train, labels_to_use, cv = 5, scoring = 'neg_mean_squared_error'))
    return(result)


# Lasso performance

# In[ ]:


score = evaluation(lasso)
print("Lasso score: {:.5f}\n".format(score.mean()))


# Ridge performance

# In[ ]:


score = evaluation(ridge)
print("Ridge score: {:.5f} \n".format(score.mean()))


# **Compare the result**
# 
# Lasso Regression give a relatively lower score than Ridge Regression. We will then adopt Lasso Regression as our best fit model. 

# In[ ]:


test_df_id = df_test_ori['Id']
test_df_x = df_test_adj.drop(['SalePrice', 'Id'], axis = 1)

lasso.fit(data_to_train, labels_to_use)
test_df_y_log = lasso.predict(test_df_x)
test_df_y = np.exp(1)**test_df_y_log

#Submission
submission = pd.DataFrame({'ID': list(test_df_id), 'SalePrice': list(test_df_y)})
submission.to_csv('submission.csv')


# **Summary**
# 
# Here come to the end of the project. 
