#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques
# 
# # Team 14
# 
# The following notebook aims to show techniques and models for predicting the Sale Prices of houses in Aimes, Iowa.
# 
# It is broken down into the following sections:
# * Explanatory Data Analysis
# * Pre-processing
#     * Handling missing values
#     * Encoding Categorical variables
# * Feature Engineering
# * Model Selection

# ## Importing Libraries and Loading the Dataset

# In[ ]:


#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# In[ ]:


sns.set(palette='dark', rc = {'figure.figsize':(8,6)})


# In[ ]:


#Importing the dataset to dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/train.csv')


# In[ ]:


# Look at the data
train.head()


# In[ ]:


# Look at the data
test.head()


# ## Exploratory Data Analysis
# 
# Our goal is to predict Sale Price, so we will start by analysing the Sale Price variable.

# In[ ]:


# Look at SalePrice

train.SalePrice.describe()


# In[ ]:


# Plot a distribution plot

plt.hist(train['SalePrice'])
plt.xlabel('Sale Price')
plt.show()


# We notice that SalePrice is skewed to the right. There aren't zero values for SalePrice and that most of the SalePrice is between 100 000 and 200 000.
# 
# We will therefore, transform SalePrice by taking its logarithm so that it is more normally distributed.
# 
# Kaggle evaluates submissions using the root mean squared log error, because of this we will take the logarithm of SalePrice now, to avoid taking the logarithm of a negative value.
# 
# Now, let's see how Sale Price looks like after transformation.

# In[ ]:


# Logarithm of SalePrice

plt.hist(np.log(train['SalePrice']))
plt.xlabel('Log SalePrice')
plt.show()


# We will now look into which variables have a linear correlation with SalePrice.

# In[ ]:


# Correlation

Corr = train.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)
Corr


# In[ ]:


#corr = Corr_05.rename(columns={'SalePrice':'Correlation'})
#corr = (corr*100).sort_values(by='Correlation')
#corr.plot(kind='barh')
#plt.xlabel('Correlation %')
#plt.ylabel('Variables')
#plt.savefig('Corr_SalePrice')
#plt.show()


# We want to look at variables that have a strong correlation with SalePrice and we find those to be OverallQual (0.79) and GrLivArea (0.71).
# 
# We also notice that the strongest correlation with SalePrice is 0.79 which is considered to be a moderate correlation and could imply that the underlying relationship is not linear as the pearson correlation describes a linear correlation between variables.

# In[ ]:


# OverallQual

train.OverallQual.describe()


# OverallQuall is a categorical variable.

# In[ ]:


# Boxplot of OverallQual and SalePrice

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = train)
plt.savefig('Overall_SalePrice')
plt.show()


# We notice that OverallQual is a categorical variable and as the OverallQual rating of a house increases, so does it's sale price.
# 
# We will look at GrLivArea now.

# In[ ]:


# Variable description

train.GrLivArea.describe()


# In[ ]:


# GrLivArea plot

train.plot(x = 'GrLivArea', y = 'SalePrice', kind='scatter')
plt.savefig('GrLiv_SalePrice')
plt.show()


# As GrLivArea increase SalePrice increases, mostly saturated below 2000 sqft.
# 
# Houses larger than 4000 are few/rare.

# ## Outliers
# 
# We will detect outliers and remove them. From the previous plot, GrLivArea looks like it contains some outliers, so we will analyse it and make use of interquatile range (IQR) to detect and remove such outliers as they will result in inaccurate approximations for our model.

# In[ ]:


# GrLivArea swarmplot

sns.swarmplot('GrLivArea', data = train)
plt.savefig('missing_gr_liv')
plt.show()


# Areas above 4000 sqft are visually represented as outlier but we will calculate the IQR and get the outlier range to be more accurate.

# In[ ]:


GrLivArea = train[['GrLivArea']]


# In[ ]:


Q1, Q2, Q3 = np.percentile(GrLivArea, [25,50,75])


# In[ ]:


IQR = Q3 - Q1


# In[ ]:


lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR


# In[ ]:


[lower_bound, upper_bound]


# In[ ]:


np.sum((train['GrLivArea'] > upper_bound) | (train['GrLivArea'] < lower_bound))


# In[ ]:


outliers = list(train[(train['GrLivArea'] < lower_bound) | (train['GrLivArea'] > upper_bound)].index)


# In[ ]:


train = train.drop(outliers)


# In[ ]:


train = train.reset_index(drop=True)


# In[ ]:


train.info()


# We have removed 31 observations that were outliers and reduced our train dataset to 1429 entries.

# ## Correlation
# 
# We'll now look at variables with a correlation greater that 0.5. Using moderate correlation as a selection criteria.

# In[ ]:


Corr_new = train.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False)
Corr_new


# We notice that our correlations have changed after handling outliers.

# In[ ]:


# Show which variables with correlation > 0.5

Corr_05 = Corr_new[Corr_new['SalePrice'] > 0.5]


# In[ ]:


# Correlation with SalePrice plot

corr = Corr_05.rename(columns={'SalePrice':'Correlation'})
corr = (corr*100).sort_values(by='Correlation')
corr.plot(kind='barh')
plt.xlabel('Correlation %')
plt.ylabel('Variables')
plt.savefig('Corr_SalePrice')
plt.show()


# ## Multi-collinearity

# In[ ]:


# Create a correlation matrix between these variables

cols = list(Corr_05.index)
Corr_map = train[cols]
Corr_map = Corr_map.corr()
Corr_map


# In[ ]:


sns.heatmap(Corr_map, vmax = 0.8, square = True, annot = True)
plt.savefig('Corrmap_')
plt.show()


# We find the following correlated pairs, with a correlation above 0.8:
# 
# * GarageCars and GarageArea
# * TotalBsmtSF and 1stFlrSF
# * TotRmsAbvGrd and GrLivArea
# 
# We will remove the column that has the lowest correlation to SalePrice

# In[ ]:


# Compare correlations

Corr_05


# In[ ]:


# Drop GarageArea, 1stFlrSF and TotRmsAbvGrd

for i in ['GarageArea', '1stFlrSF',]: # 'TotRmsAbvGrd'
    cols.remove(i)
cols


# We will run a regression on SalePrice with these variables once we've dealt with missing values and categorical variables.

# In[ ]:


# Look at the remaining columns

sns.pairplot(train[cols], size=3)
#plt.savefig('pairplot.png')
plt.show()


# ## Qualitative and Quantitative

# We will create a numeric and categorical dataframe, to handle such variables seperately.

# In[ ]:


# Create numeric features

train_numeric_features = train.select_dtypes([np.number])
test_numeric_features = test.select_dtypes([np.number])


# We will drop Id because its not a feature but an observation identifier and will not be a part of our model.

# In[ ]:


# Drop Id column

train_numeric_features = train_numeric_features.drop('Id', axis=1)
test_numeric_features = test_numeric_features.drop('Id', axis=1)


# In[ ]:


# Create categorical features

categorical_features_train = train.drop(list(train_numeric_features.columns), axis=1)
categorical_features_test = test.drop(list(test_numeric_features.columns), axis=1)


# In[ ]:


categorical_features_train.info()


# ## Handling missing values

# Check for numerical and categorical missing values

# In[ ]:


def missing_plot(df):
    
    '''df = pandas dataframe
    
    Calculates missing values in a dataframe and plots a horizontal bar plot of the results
    '''
    
    missing = ((df.isnull().sum() / len(df))*100).sort_values()
    missing = missing.to_frame('missing_percentage')
    missing = missing[missing['missing_percentage'] > 0]
    missing.plot(kind='barh', figsize=(7,5))
    plt.xlabel('Percentage missing')
    plt.ylabel('Variables')
    plt.show()
    return 


# In[ ]:


# Numeric train data missing values

missing_plot(train_numeric_features)


# There's missing data in 3 variables: LotFrontage, GarageYrBlt, MasVnrArea, with atmost 17.7% of the data missing.

# In[ ]:


# Numeric test data missing values

missing_plot(test_numeric_features)


# There's missing data in 11 variables: LotFrontage, GarageYrBlt, MasVnrArea, BsmtHalfBath, BsmtFullBath, BsmtFinSF1, GarageArea, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, GarageCars,  with atmost 15.6% of the data missing.

# In[ ]:


# Categorical train data missing values

missing_plot(categorical_features_train)


# In[ ]:


# Categorical test data missing values

missing_plot(categorical_features_test)


# The categorical variables seem to have a lot of missing data with PoolQC, MiscFeature and Alley having over 90% of the data missing.
# 
# From the documentation we deduce that we should replace these 'NaN' values to string 'None' since for categorical features they mean the feature isn't present and not missing value.

# In[ ]:


# Identify columns to transform

misc_feature = ['MiscFeature']
Cat_None = ['Fence', 'PoolQC', 'GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'FireplaceQu'             , 'BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual'             , 'Alley', 'MiscFeature']


# In[ ]:


#categorical_features_train[misc_feature + Cat_None].head(10)


# In[ ]:


# Transform variables

categorical_features_train[misc_feature + Cat_None] = categorical_features_train[misc_feature + Cat_None]                                                        .replace(np.nan, 'None')
categorical_features_test[misc_feature + Cat_None] = categorical_features_test[misc_feature + Cat_None]                                                        .replace(np.nan, 'None')


# Check how many missing values are left now

# In[ ]:


# Categorical train data missing values after 'None' replacing

missing_plot(categorical_features_train)


# In[ ]:


# Categorical test data missing values after 'None' replacing

missing_plot(categorical_features_test)


# There are still missing values left, so we'll handle those by filling them with their variable's mode.

# In[ ]:


# Fill using the mode given the Neighbourhood

categorical_features_train = categorical_features_train.apply( lambda x : x.fillna(x.mode().loc[0]) )
categorical_features_test = categorical_features_test.apply( lambda x : x.fillna(x.mode().loc[0]) )


# Lets see if the missing values have been filled

# In[ ]:


(categorical_features_train.isnull().sum() / len(categorical_features_train)).sort_values(ascending=False)


# In[ ]:


(categorical_features_test.isnull().sum() / len(categorical_features_test)).sort_values(ascending=False)


# All categorical missing values have been handled.

# We will now handle numerical missing values

# In[ ]:


train_numeric_features = train_numeric_features.fillna(train_numeric_features.mean())
test_numeric_features = test_numeric_features.fillna(test_numeric_features.mean())


# Check if the missing values are filled

# In[ ]:


(train_numeric_features.isnull().sum() / len(train_numeric_features)).sort_values(ascending=False)


# In[ ]:


(test_numeric_features.isnull().sum() / len(test_numeric_features)).sort_values(ascending=False)


# All missing values have been handled. We will now check for variables that refer to years and ensure their data type is 'int64'.

# In[ ]:


# Check data types

train_numeric_features.select_dtypes(['float64']).columns


# In[ ]:


# Check data types

test_numeric_features.select_dtypes(['float64']).columns


# We will change GarageYrBlt to 'int64' type.

# In[ ]:


# Change data type

train_numeric_features['GarageYrBlt'] = train_numeric_features['GarageYrBlt'].astype('int64')
train_numeric_features.select_dtypes(['float64']).columns


# In[ ]:


test_numeric_features['GarageYrBlt'] = test_numeric_features['GarageYrBlt'].astype('int64')
test_numeric_features.select_dtypes(['float64']).columns


# Data types have been handled

# ## Encode Categorical Features
# 
# We will join our categorical and numeric dataframes before we encode for categorical variables, so that our dummy variables align between our train and test dataset.

# In[ ]:


# Join categorical and numerical dataframes

data_train = pd.concat([train_numeric_features, categorical_features_train], axis=1)
data_test = pd.concat([test_numeric_features, categorical_features_test], axis=1)


# In[ ]:


# Join train and test dataset so encoding can align

data = data_train.append(data_test, ignore_index=True)


# In[ ]:


# Creating dummy variables

df = data.copy()
for i in list(categorical_features_train.columns): #
    dummy = pd.get_dummies(data = df[[i]], drop_first = True)
    df = pd.concat([df, dummy], axis=1).drop(i, axis=1)


# In[ ]:


df.info() # no object dtypes


# After creating dummy variables, we now have 260 columns.

# ## Multiple Linear Regression

# We are going to start with a multiple linear regression with the columns 'cols' we selected earlier as a starting point.
# 
# With the criteria we used we now only have 8 independent variables for our model.
# 
# Since, we have succesfully encoded our categorical variables, will split our data into train and test as before.

# In[ ]:


# Seperate Train and Test set

train_df = df.loc[:1428]
test_df = df.loc[1429:].drop(['SalePrice'], axis=1)


# We are going to take log of SalePrice for our analysis because we'll be using mean squared log error to calculate our residual error and taking logs now avoids returning an error if we try take logs of negative values.

# In[ ]:


# Define X and y variables

X_1 = train_df[cols].drop('SalePrice', axis=1).values
y_1 = np.log(train_df[cols][['SalePrice']].values)


# We will now split the data into a training set we'll use to train our model and a test set for testing the model. We will use 2/3 of the data for training and 1/3 for testing.

# In[ ]:


X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y_1, test_size = 0.33, random_state = 42)


# We will now fit the training set to the model and predict oury values.

# In[ ]:


# Fit training data to linear model

lm_1 = LinearRegression()
lm_1 = lm_1.fit(X_1_train, y_1_train)
y_1_predict = lm_1.predict(X_1_test)


# In[ ]:


lm_1.intercept_


# In[ ]:


Coeff_1 = pd.DataFrame(lm_1.coef_[0], train_df[cols].drop('SalePrice', axis=1).columns, columns = ['Coefficients'])
Coeff_1.sort_values(by='Coefficients', ascending = False)


# In[ ]:


print('Model 1 Train R2:', metrics.r2_score(y_1_train, lm_1.predict(X_1_train)))


# In[ ]:


print('Model 1 Test R2:', metrics.r2_score(y_1_test, y_1_predict))


# In[ ]:


print('Model 1 Train RMSLE:', np.sqrt(metrics.mean_squared_error(y_1_train, lm_1.predict(X_1_train))))


# In[ ]:


print('Model 1 Test RMSLE:', np.sqrt(metrics.mean_squared_error(y_1_test, y_1_predict)))


# We will now plot the results

# In[ ]:


# Predicted vs Actual Plot

plt.scatter(np.exp(y_1_predict), np.exp(y_1_test), alpha = 0.4)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Actual SalePrice')
plt.show()


# In[ ]:


# Residual Plot

plt.scatter(y_1_predict, y_1_test - y_1_predict, alpha = 0.4)
#plt.plot(x = np.arange(len(y_1_predict)), y = np.zeros(y_1_predict.shape), color = 'k', marker = '--')
plt.xlabel('Predicted Log SalePrice')
plt.ylabel('Log Residuals')
plt.show()


# Our first model is performing relatively good, over 80% of the variation in our target variable is explain by the independent variables. There is a small increase in the test RMSLE which is expected and it doesn't deviate excessively.
# 
# This would imply our model behaves well under unseen data but because we only used linear correlation as our criteria, we will now use backward elimination with a significance level of 0.05 which should be a stronger variable selection criteria.

# In[ ]:


# Submission

X_sub = test_df[cols[1:]].values
#scaler_sub = StandardScaler()
#X_sub_scalered = scaler_sub.fit_transform(X_sub)
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = np.exp(lm_1.predict(X_sub))
submission.to_csv('submission_a.csv', index=False)
submission.head()


# ## Backward Elimination

# In[ ]:


# Define X and y variables for training data

X_2 = train_df.drop(['SalePrice'], axis=1).values
y_2 = np.log(train_df[['SalePrice']].values)


# The functions OLS and BackwardElimination below have been sourced from the Udemy course Machine Learning: A-Z with minor modifications.
# 
# We expanded this function to be able to extract the eliminated variables and perform a linear regression.

# In[ ]:


# OLS Summary

def OLS(X, y):
    import statsmodels.formula.api as sm
    X_opt = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis=1)
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    return regressor_OLS.summary()


# In[ ]:


OLS(X_2, y_2)


# Without dropping any variables we find an $R^2$ of 94.4%, most likely because of too many variables we have overfit our model, so will use backward elimination to reduce the variables.

# In[ ]:


# Create X with intercept variable

X_opt = np.append(arr = np.ones((len(X_2),1)).astype(int), values = X_2, axis=1)


# In[ ]:


# Variable selection using backward elimination

def Backward_Elimination(x, y, sl):
    import statsmodels.formula.api as sm
    del_var_ind = []
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    del_var_ind.append(j)
    regressor_OLS.summary()
    return x, del_var_ind


# In[ ]:


# Return Optimal Model X and the dropped variables with a p-value less than 5%

opt_X, del_var = Backward_Elimination(X_opt, y_1, 0.05)


# In[ ]:


len(del_var)


# Backward elimination has deleted 171 variables

# In[ ]:


# Get columns list excluding SalePrice

num_col = df.drop('SalePrice', axis=1).columns

# Map to variable name

var = {}
for i in range(len(list(num_col))):
    var.update({i+1:num_col[i]})

# List of optimal model columns index

opt_col_ind = []
for i in list(var.keys()):
    if i not in del_var:
        opt_col_ind.append(i)
        
# List of optimal model columns variable names

opt_col = []
for i in opt_col_ind:
    opt_col.append(var.get(i))
len(opt_col)


# We have reduced our variables to 152 independent variables

# We will now run a linear regression on these variables only and compare error with previous model
#     
# We are going to remain consistent and have a 80% of the data for our training set and 20% for the testing set.

# In[ ]:


# Defining X under backward elimination resulted variables

X_2 = train_df.drop(['SalePrice'], axis=1)[opt_col].values
X_test_df_2 = test_df[opt_col].values


# In[ ]:


# Split between train and test
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.33, random_state=42)


# In[ ]:


lm_2 = LinearRegression()
lm_2.fit(X_2_train, y_2_train)


# In[ ]:


y_2_pred = lm_2.predict(X_2_test)


# In[ ]:


lm_2.intercept_


# In[ ]:


Coeff_2 = pd.DataFrame(lm_2.coef_[0], train_df.drop(['SalePrice'], axis=1)[opt_col].columns, columns = ['Coefficients'])
Coeff_2.sort_values(by='Coefficients', ascending = False)


# In[ ]:


# Training R squared

print('Model 2 Train R2:', metrics.r2_score(y_2_train, lm_2.predict(X_2_train)))


# In[ ]:


# Testing R squared

print('Model 2 Test R2:', metrics.r2_score(y_2_test, y_2_pred))


# In[ ]:


# Training RMSLE

print('Model 2 Training RMSLE:', np.sqrt(metrics.mean_squared_error(y_2_train, lm_2.predict(X_2_train))))


# In[ ]:


# Testing RMSLE

print('Model 2 Testing RMSLE:', np.sqrt(metrics.mean_squared_error(y_2_test, y_2_pred)))


# We will now plot these results

# In[ ]:


# Predicted vs Actual Plot

plt.scatter(np.exp(y_2_pred), np.exp(y_2_test), alpha = 0.4)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Actual SalePrice')
plt.show()


# In[ ]:


# Residual Plot

plt.scatter(y_2_pred, y_2_test - y_2_pred, alpha = 0.4)
plt.xlabel('Predicted Log SalePrice')
plt.ylabel('Log Residuals')
plt.show()


# Our test set error has increased and $R^2$ has decreased. They deviate much further in comparison to the previous model.
# 
# We will try another model and see how it performs in comparison and if we can improve on the test error.

# In[ ]:


# Submission

X_sub = test_df[opt_col].values
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = np.exp(lm_2.predict(X_sub))
submission.to_csv('submission_b.csv', index=False)
submission.head()


# ## Ridge Regression
# 
# Since ridge regression minimises the squared residual error and independent variables coefficient, we will firstly scalar the independent variables so that they can be comparable. We use standardisation, which subtracts the mean of the variable from each observation and devides it by its standard deviation.

# In[ ]:


X_3 = train_df.drop(['SalePrice'], axis=1)
y_3 = np.log(train_df[['SalePrice']])


# In[ ]:


scaler_1 = StandardScaler()


# In[ ]:


X_3_scaled = scaler_1.fit_transform(X_3)


# In[ ]:


X_standardize = pd.DataFrame(X_3_scaled,columns=X_3.columns)
X_standardize.head()


# Our variables have been standardized.
# 
# We will once again, split our data into training and testing set using the same parameters.

# In[ ]:


X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(X_standardize, 
                                                    y_1, 
                                                    test_size=0.33, 
                                                    random_state = 42)


# We will now fit a simple ridge regression with alpha being 1. 

# In[ ]:


ridge = Ridge(random_state=42)


# In[ ]:


ridge.fit(X_3_train, y_3_train)


# In[ ]:


y_3_pred = ridge.predict(X_3_test)


# In[ ]:


ridge.intercept_


# In[ ]:


Coeff_2 = pd.DataFrame(ridge.coef_[0], train_df.drop(['SalePrice'], axis=1).columns, columns = ['Coefficients'])
Coeff_2.sort_values(by='Coefficients', ascending = False)


# In[ ]:


# Training R squared

print('Model 2 Train R2:', metrics.r2_score(y_3_train, ridge.predict(X_3_train)))


# In[ ]:


# Testing R squared

print('Model 2 Test R2:', metrics.r2_score(y_3_test, y_3_pred))


# In[ ]:


# Training RMSLE

print('Model 2 Training RMSLE:', np.sqrt(metrics.mean_squared_error(y_3_train, ridge.predict(X_3_train))))


# In[ ]:


# Testing RMSLE

print('Model 2 Testing RMSLE:', np.sqrt(metrics.mean_squared_error(y_3_test, y_3_pred)))


# In[ ]:


# Predicted vs Actual Plot

plt.scatter(np.exp(y_3_pred), np.exp(y_3_test), alpha = 0.4)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Actual SalePrice')
plt.show()


# In[ ]:


# Residuals Plot

plt.scatter(y_3_pred, y_3_test - y_3_pred, alpha = 0.4)
plt.xlabel('Predicted Log SalePrice')
plt.ylabel('Log Residuals')
plt.show()


# This model doen't perform well as the $R^2$ increases by almost 8% from training to testing set and the error about doubles. This is because we chose a weak alpha of 1, we will improve on this by finding an alpha that minimises the error and coefficients.

# In[ ]:


# Submission

X_sub = test_df.values
scaler_sub = StandardScaler()
X_sub_scalered = scaler_sub.fit_transform(X_sub)
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = np.exp(ridge.predict(X_sub_scalered))
submission.to_csv('submission_c.csv', index=False)
submission.head()


# ## Ridge Regression with Cross Validation
# 
# There is empirical evidence (https://machinelearningmastery.com/k-fold-cross-validation/) that a 10-fold cross validation produces estimates with low bias and variance. 
# We weill therefore use a 10-fold cross validation to find our optimal alpha, and then perform a ridge regularization on our variables with that alpha. 

# In[ ]:


alphas = np.arange(0,105,5)
ridge_cv = RidgeCV(alphas=alphas, cv=10) # using 10 fold validation methos


# In[ ]:


ridge_cv.fit(X_3_train, y_3_train)


# In[ ]:


ridge_cv.alpha_


# Cross validation has picked an alpha of 100

# In[ ]:


y_3_cv_pred = ridge_cv.predict(X_3_test)


# Lets look at the coefficients of this model.

# In[ ]:


Coeff_3 = pd.DataFrame(ridge_cv.coef_[0], train_df.drop(['SalePrice'], axis=1).columns, columns = ['Coefficients'])
Coeff_3.sort_values(by='Coefficients', ascending = False)


# Our coefficients have decreased with the largest one being lower that 0.06

# In[ ]:


# Training R squared

print('Model 3 Train R2:', metrics.r2_score(y_3_train, ridge_cv.predict(X_3_train)))


# In[ ]:


# Testing R squared

print('Model 3 Test R2:', metrics.r2_score(y_3_test, y_3_cv_pred))


# In[ ]:


# Training RMSLE

print('Model 3 Training RMSLE:', np.sqrt(metrics.mean_squared_error(y_3_train, ridge_cv.predict(X_3_train))))


# In[ ]:


# Testing RMSLE

print('Model 3 Testing RMSLE:', np.sqrt(metrics.mean_squared_error(y_3_test, y_3_cv_pred)))


# Lets plot these results

# In[ ]:


# Predicted vs Actual Plot

plt.scatter(np.exp(y_3_cv_pred), np.exp(y_3_test), alpha = 0.4)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Actual SalePrice')
plt.show()


# In[ ]:


# Residuals Plot

plt.scatter(y_3_cv_pred, y_3_test - y_3_cv_pred, alpha = 0.4)
plt.xlabel('Predicted Log SalePrice')
plt.ylabel('Log Residuals')
plt.show()


# This model has improved from the ridge with alpha at 1 and our test error has decresed in comparison to the model with alpha at 1. 

# In[ ]:


# Submission

X_sub = test_df.values
scaler_sub = StandardScaler()
X_sub_scalered = scaler_sub.fit_transform(X_sub)
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = np.exp(ridge_cv.predict(X_sub_scalered))
submission.to_csv('submission_d.csv', index=False)
submission.head()


# ## Lasso Regression
# 
# The lasso regression work similar to ridge regression, differing by minimising the absolute values of the coefficients. We use cross validation to find the optimal alpha.

# In[ ]:


alphas = np.arange(0.01,0.11,0.001)
lasso_cv = LassoCV(alphas=alphas, cv=10)# using 10 fold validation methos
lasso_cv.fit(X_3_train, y_3_train)


# In[ ]:


lasso_cv.alpha_


# Our optimal alpha is 0.01

# In[ ]:


y_lasso = lasso_cv.predict(X_3_test)


# In[ ]:


lasso_cv.intercept_


# In[ ]:


Coeff_4 = pd.DataFrame(lasso_cv.coef_, train_df.drop(['SalePrice'], axis=1).columns, columns = ['Coefficients'])
Coeff_4.sort_values(by='Coefficients', ascending = False)


# In[ ]:


# Training R squared

print('Model 4 Train R2:', metrics.r2_score(y_3_train, lasso_cv.predict(X_3_train)))


# In[ ]:


# Testing R squared

print('Model 4 Test R2:', metrics.r2_score(y_3_test, y_lasso))


# In[ ]:


# Training RMSLE

print('Model 4 Training RMSLE:', np.sqrt(metrics.mean_squared_error(y_3_train, lasso_cv.predict(X_3_train))))


# In[ ]:


# Testing RMSLE

print('Model 4 Testing RMSLE:', np.sqrt(metrics.mean_squared_error(y_3_test, y_lasso)))


# Lets plot our model results

# In[ ]:


# Predicted vs Actual Plot

plt.scatter(np.exp(y_lasso), np.exp(y_3_test), alpha = 0.4)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Actual SalePrice')
plt.show()


# In[ ]:


# Residuals Plot

plt.scatter(y_lasso, y_3_test.squeeze() - y_lasso, alpha = 0.4)
plt.xlabel('Predicted Log SalePrice')
plt.ylabel('Log Residuals')
plt.show()


# 

# In[ ]:


# Submission

X_sub = test_df.values
scaler_sub = StandardScaler()
X_sub_scalered = scaler_sub.fit_transform(X_sub)
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = np.exp(lasso_cv.predict(X_sub_scalered))
submission.to_csv('submission_e.csv', index=False)
submission.head()


# ## Conclusion
# 
# The Lasso Regression model perfoms the best with the lowest RMSLE for the testing set and the least difference in error between traing and testing set. It also has the highest $R^2$ for the testing set with the least difference between training and testing set.
