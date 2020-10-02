#!/usr/bin/env python
# coding: utf-8

# # EDSA JHB T20 
# ## Regression Sprint on the Ames, Iowa housing dataset
# ### Katleho Khoali, Keshav Chetty, Riaan Swanepoel, Siphephelo Gcabashe, Tshepo Molope
# 
# This is team 20's kernel for the House Prices: Advanced Regression Techniques Kaggle competition for the EDSA Regression sprint. In this kernel we will explore the Ames, Iowa housing dataset by performing some exploratory data analysis and performing preprocessing on the data as requires. There after we will do some feature engineering before we build various models to try to predict the sales price of houses on a test dataset.

# ## Importing libraries
# We import the usual suspects as well as various scikit-learn libraries that will be required for our model

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import skew

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings('ignore')


# ## Creating functions required for later use
# Here we create a couple of functions that will be used later

# In[ ]:


#These functions will be used to assess the performance of the models
def rmse_cv(model, X, y):
    '''Calculates the cross validation rmse of the prediction
    
    Parameters
    model (sklearn model): The model for which the cross validation should be performed
    X (Pandas dataframe): The dataframe that the model should perform the predictions on
    y (Series): The values that the prediction should be compared against
    '''
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)
  

def rmsle (y, y_pred):
    '''Calculates the rmsle of the prediction
    
    Parameters
    y (Series): The target values to assess the prediction aganst
    y_pred (Series): The predicted values to assess
    '''
    return (np.sqrt(metrics.mean_squared_error(y, y_pred)))

  
def CVscore(model, X, y):
    '''Calculates the cross validation score of the prediction
    
    Parameters
    model (sklearn model): The model for which the cross validation should be performed
    X (Pandas dataframe): The dataframe that the model should perform the predictions on
    y (Series): The values that the prediction should be compared against
    '''
    result = cross_val_score(model, X, y, cv=kfold)
    return result.mean()
  
  
#This function will be used to combine different models into an ensembled model  
def blend_models_predict(X, models):
    '''Takes list of models and returns a blended set of the models
    
    Parameters
    X (Pandas dataframe): The dataframe that the model should perform the predictions on
    models (list of sklearn pipelines): A list of model pipelines for which the predictions will be blended
    '''
    blend = []
    for model in models:
        blend.append(1/len(models)*model.predict(X))
    return sum(blend)


# ## Importing data
# We import the train and test datasets and store them as pandas dataframes

# In[ ]:


df_train = pd.read_csv('../input/train.csv',index_col='Id')
df_test = pd.read_csv('../input/test.csv',index_col='Id')


# ## Identifying outliers
# It is recommended to identify and remove outlies from the training set which should allow the model to fit the dataset better.

# The author of the Ames housing dataset recommends removing all datapoints with a 'GrLivArea' > 4000. However, we decided to only remove the outliers where the 'GrLivArea' > 4000 and the 'SalePrice' < 300000 since those with a high 'GrLivArea' and high 'SalePrice' might be an indication of the true 'SalePrice' for a house of that size.
# 
# Ref: http://jse.amstat.org/v19n3/decock.pdf

# In[ ]:


sns.scatterplot(df_train['GrLivArea'], df_train['SalePrice'])
plt.title('Identifying outliers')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')


# In[ ]:


df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, inplace=True)


# In[ ]:


#This variable is used to keep track of the training dataset when we combine it with the test dataset
len_train = len(df_train)


# In[ ]:


df_train.head()


# ## Setting the target variable
# We assign the target variable to y, which we will first explore and later we will use this variable to train our models

# In[ ]:


y = df_train['SalePrice']


# In[ ]:


y.describe()


# In[ ]:


sns.distplot(y, fit=st.norm)


# From this we can see that SalePrice does not follow a normal distribution and is positively skew. We therefore need to transform SalePrice in order to meet several statistical asumptions about the data as well as to avoid Homoscedasticity (The response variable has the same standard deviation regardless of the predictor variables), which may over estimate the goodness of fit. 

# In[ ]:


#Log transformation is one of the simplest transformations to normalise positively skewed data
y = np.log1p(df_train['SalePrice'])
sns.distplot(y, fit=st.norm)


# ## Combining the testing and training dataset in order to more easily preprocess the datasets more effectively

# The test and train set is combined in order to manipulate both datasets simultaniously and in the exact same way. A previous variable (len_train) was created to keep track of the location of the train dataset in the combined set.

# In[ ]:


df_all = pd.concat([df_train, df_test], sort=False)


# ## Exploring the data

# In[ ]:


print(df_all.shape)
print(df_all.info())


# From the info of the dataset, we see that there is a combination of numerical and categorical variables. These will have to be explored further and possibly seperately in order to identify any problems within the dataset

# ## 4C's of data cleaning
# From https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# 
# Data cleaning usually consists of four broad steps: 
# 
# * Correcting 
# * Completing 
# * Converting
# * Creating
# 
# 

# ### Correcting
# 
# Reviewing the data is required to ensure there are no aberrant or unacceptable data points. For example, if a categorical variable like Neighborhood has a spelling error in one of the entries, the model will see that as two seperate categories and it will affect the prediction. In the case of numerical variables, data types which logically should be integers like years, can't have float values. However, caution should be used when changing data as those could be the actual entries. It is therefore required to read the data descriptions and have some domain knowledge before making changes to the data.

# In[ ]:


df_all['Electrical'].unique()


# In[ ]:


df_all['Neighborhood'].unique()


# Based on the data descriptions, there does not seem to be any entries that needs correcting.

# ### Completing
# 
# There are often null values in datasets which occurs when certain entries cannot be measured or entered. Some algorithms do not know how to deal with null values or missing and these values should therefore be filled in with what makes sense. Alley for example has large amounts of missing values, however these missing values just mean that the particular house does not have an alley, it is therefore recommended to fill the missing values in with the string 'None' in order to not lose the value of these datapoints. Other variables can be filled in with a 0 or the median or mean of the variable. It is important to read the data descriptions and have some domain knowledge to determine the best way to fill in missing values.

# In[ ]:


#Check number of missing values
missing = df_all.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

#Note that SalePrice also have missing values, but this is due to the absence of SalePrice from the test dataset. We will therefore not fill in the SalePrice


# In the graph above, we can see that there is quite a lot of features with missing values. While the data descriptions do give us an indication of what the missing values mean for some of the features, others require intuition and an understanding of the data to determine what to fill the missing values with. 
# 
# The following kernel gives a handy table that shows which features are which data type and what makes most sense to fill in the missing values with: https://www.kaggle.com/fugacity/ridge-lasso-elasticnet-regressions-explained

# In[ ]:


#The following features are categorical features that needs to be filled in with 'None'
Cat_toNone = ('PoolQC','Fence','MiscFeature','Alley','FireplaceQu','GarageType','GarageFinish',
              'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
              'BsmtFinType2','MasVnrType','MSSubClass')

#This loop takes the tuple of features and fills all the missing values with 'None'
for c in Cat_toNone:
    df_all[c] = df_all[c].fillna('None')


# In[ ]:


#The following features are categorical features that needs to be filled in with the mode
Cat_toMode = ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd',
              'Utilities','SaleType','Functional')

#This loop takes the tuple of features and fills all the missing values with the mode
for c in Cat_toMode:
    df_all[c] = df_all[c].fillna(df_all[c].mode()[0])


# In[ ]:


#The following features are numerical features that needs to be filled in with 0
Num_toZero = ('GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
              'BsmtFullBath','BsmtHalfBath','GarageYrBlt','MasVnrArea')

#This loop takes the tuple of features and fills all the missing values with 0
for c in Num_toZero:
    df_all[c] = df_all[c].fillna(0)


# In[ ]:


#In this case, neigborhood might have a large influence on the lot frontage. Therefore we group by Neighborhood and fill the LotFrontage with the median of the groups
df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[ ]:


#Recheck number of missing values to identify any missed features
missing = df_all.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

#Note that SalePrice also have missing values, but this is due to the absence of SalePrice from the test dataset


# ### Converting
# 
# Here we will convert features into the correct data types. MSSubClass for example is actually a categorical variable, but Python sees it as a numerical variable. Therefore we convert the variable into a string in order for the algorithms to treat it as a categorical variable.

# In[ ]:


df_all['MSSubClass'] = df_all['MSSubClass'].astype(str)


# In[ ]:


#We make a list of all the numerical values here in order to exclude the numerical variables from the ordinal variables that we will create in the next step
numeric = df_all.select_dtypes(exclude = ['object']).columns


# In[ ]:


df_all.hist(figsize = (12, 7), bins = 40)
plt.tight_layout()
plt.show()


# The bargraphs of the numerical features show us that there is a couple of skewed numerical features. We therefore identify the features with a skweness of more than 0.75 which we will log transform.

# In[ ]:


skewed_feats = df_all[numeric].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

df_all[skewed_feats] = np.log1p(df_all[skewed_feats])

#df_all[numeric] = np.log1p(df_all[numeric])


# In[ ]:


df_all.hist(figsize = (12, 7), bins = 40)
plt.tight_layout()
plt.show()


# Many of the skewed features have now been transformed and is now normally distributed

# Certain categorical variables are ordinal, for example Quality can be measured on a scale of 1 to 5 with 1 being none poor quality and 5 being excelent quality. Therefore we convert the categorical variables that are ordinal into an ordered set of integers.

# In[ ]:


#We replace all ordinal categorical variables with an ordered set of integers
df_all = df_all.replace({"Alley" : {"None" : 0, "Grvl" : 1, "Pave" : 2},
                     "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                     "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                       "ALQ" : 5, "GLQ" : 6},
                     "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                       "ALQ" : 5, "GLQ" : 6},
                     "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                     "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                     "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                     "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                     "Min2" : 6, "Min1" : 7, "Typ" : 8},
                     "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                     "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                     "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                     "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                     "Street" : {"Grvl" : 1, "Pave" : 2},
                     "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                    )


# ### Creating
# 
# We use feature engineering to create new features that makes sense from existing features. this could in certain cases provide new signals which could improve the predictions. Here we will create various predictors like the total number of Bathrooms and HouseAge

# In[ ]:


#Combining bathrooms into single bathroom feature
df_all['Bathrooms'] = df_all['FullBath'] + (df_all['HalfBath']*0.5) + df_all['BsmtFullBath'] + (df_all['BsmtHalfBath']*0.5)

#Combining fireplace and fireplace quality into a fireplace score
df_all['FireplaceScore'] = df_all['Fireplaces'] * df_all['FireplaceQu']

#Creating a single GarageScore
#GarageCond does not add to the correlation, and is therefore excluded
df_all['GarageScore'] = df_all['GarageCars'] * df_all['GarageQual'] * df_all['GarageArea']

#Determining house age
df_all['HouseAge'] = df_all['YrSold'] - df_all['YearBuilt']

#Total Living square feet
df_all['TotalLivingSF'] = df_all['GrLivArea'] + df_all['TotalBsmtSF'] - df_all['LowQualFinSF']


# ## Identifying important features relative to the target
# Next we perform a correlation analysis between the features to identify the features highly correlated with the SalePrice, as well as the features that have high collinearity with other features.

# In[ ]:


print("Find most important features relative to target")
corr = df_all.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, cmap='coolwarm')
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)


# ## Remove unnecessary features
# Since we created some features from existing features, we remove these features as well as any other highly collinear features.

# In[ ]:


df_all = df_all.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
                      'GarageCars','GarageArea','GarageQual','GarageCond',
                      'FireplaceQu','Fireplaces', 
                      'PoolArea', #Pool Quality is more important than pool size
                      'Utilities', 'Street',
                      'GarageYrBlt'
                     ], axis = 1)


# ## Create dummy variables for categorical features
# Most linear regression algorithms can't use categorical data for predictions, thus we  have to create dummy variables for the categorical features. We drop the first dummy feature for each categorical feature in order to avoid falling into the dummy variable trap where one  dummy feature is completely collinear to all the other dummy features.

# In[ ]:


df_all = pd.get_dummies(df_all, drop_first=True)


# ## Splitting the training and testing dataset up again
# After we've done all the data preprocessing and cleaning on the combines dataset, we have to split it up again into the train and test set. We use the len_train variable that we created previously to keep track of the train dataset to split the combined dataset 

# In[ ]:


df_all = df_all.drop(['SalePrice'], axis = 1)
train = df_all[:len_train]
test = df_all[len_train:]


# ## Split the training set up using train_test_split
# Here we split the training set up into datasets that we can build and test our model on

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)


# ## Modeling
# There are multiple different algorithms to perform linear regression. The three most commonly used is Multiple linear regression (this also includes regularization with Ridge, Lasso and ElasticNet), Decision tree regression and Random forest regression.
# In this kernel we create models using all of these common regression algorithms in order to determine the best algorithm for this dataset.

# In[ ]:


#Multiple Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#Predicting the Test set results
train_lm = regressor.predict(X_train)
test_lm = regressor.predict(X_test)

#Multiple Linear Regression using RIDGE regularization
ridge = Ridge(alpha = 10)
ridge.fit(X_train, y_train)
#Predicting the Test set results
ridge_train_lm = ridge.predict(X_train)
ridge_test_lm = ridge.predict(X_test)

#Multiple Linear Regression using LASSO regularization
lasso = Lasso(max_iter=500,alpha = 0.001)
lasso.fit(X_train, y_train)
#Predicting the Test set results
lasso_train_lm = lasso.predict(X_train)
lasso_test_lm = lasso.predict(X_test)

#https://www.kaggle.com/fugacity/ridge-lasso-elasticnet-regressions-explained
#Multiple Linear Regression using ELASTICNET regularization
en = ElasticNet(max_iter=500,alpha = 0.001)
en.fit(X_train, y_train)
#Predicting the Test set results
en_train_lm = en.predict(X_train)
en_test_lm = en.predict(X_test)

#Decision Tree Regression
dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(X_train, y_train)
#Predicting the Test set results
train_dtr = dtr.predict(X_train)
test_dtr = dtr.predict(X_test)

#Random Forest Regression
rf = RandomForestRegressor(random_state = 0)
rf.fit(X_train, y_train)
#Predicting the Test set results
train_rf = rf.predict(X_train)
test_rf = rf.predict(X_test)


# ### Assessing models
# We will now assess our models based on the RMSE on the testing and training set, as well as the RMSLE between the predicted SalePrice and the true SalePrice

# In[ ]:


scorer = metrics.make_scorer(metrics.mean_squared_error, greater_is_better = False)
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


# In[ ]:


#Assessing the Linear Regression model 
print('Linear Regression')
print("RMSE on Training set :", rmse_cv(regressor,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(regressor,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, regressor.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, regressor.predict(X_test)))
print('-'*25)

#Assessing the Ridge Linear Regression model 
print('Linear Regression Ridge')
print("RMSE on Training set :", rmse_cv(ridge,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(ridge,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, ridge.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, ridge.predict(X_test)))
print('-'*25)

#Assessing the Lasso Linear Regression model 
print('Linear Regression Lasso')
print("RMSE on Training set :", rmse_cv(lasso,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(lasso,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, lasso.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, lasso.predict(X_test)))
print('-'*25)

#Assessing the ElasticNet Linear Regression model 
print('Linear Regression ElasticNet')
print("RMSE on Training set :", rmse_cv(en,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(en,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, en.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, en.predict(X_test)))
print('-'*25)

#Assessing the Decision Tree model 
print('Decision Tree Regression')
print("RMSE on Training set :", rmse_cv(dtr,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(dtr,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, dtr.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, dtr.predict(X_test)))
print('-'*25)

#Assessing the Random Forest model 
print('Random Forest Regression')
print("RMSE on Training set :", rmse_cv(rf,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(rf,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, rf.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, rf.predict(X_test)))


# These results show that the Multiple Linear Regression using the various regularization methods out performed normal Multiple Linear Regression as well as both the Decision Tree and Random Forest, therefore we decided not to use the normal linear regression, decision tree nor the random forest for our model.
# 
# It should be noted that this was using a single arbitrary alpha parameter. Can we further improve the score using hyperparameter selection to identify and use the best alpha parameter for each method?

# ### Hyperparameter selection
# During hyperparameter selection we give the methods a range of alpha values. The model uses these alpha values, makes predictions and assess the rmse of each alpha parameter. The alpha with the lowest rmse, is the best alpha parameter for the particular method.

# In[ ]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha),X_train,y_train).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
print('Best alpha parameter: ', cv_ridge[cv_ridge == min(cv_ridge)].index[0])


# In[ ]:


alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012]
cv_lasso = [rmse_cv(Lasso(max_iter=500, alpha = alpha),X_train,y_train).mean() 
            for alpha in alphas]

cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
print('Best alpha parameter: ', cv_lasso[cv_lasso == min(cv_lasso)].index[0])


# In[ ]:


alphas = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002]
cv_elasticnet = [rmse_cv(ElasticNet(max_iter=500,alpha = alpha),X_train,y_train).mean() 
            for alpha in alphas]

cv_elasticnet = pd.Series(cv_elasticnet, index = alphas)
cv_elasticnet.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")
print('Best alpha parameter: ', cv_elasticnet[cv_elasticnet == min(cv_elasticnet)].index[0])


# We can see that each model has it's own best alpha parameter. Lets see how the optimal alpha parameter affects the predictions.

# In[ ]:


#Multiple Linear Regression using RIDGE regularization
ridge = Ridge(alpha = cv_ridge[cv_ridge == min(cv_ridge)].index[0])
ridge.fit(X_train, y_train)
#Predicting the Test set results
ridge_train_lm = ridge.predict(X_train)
ridge_test_lm = ridge.predict(X_test)

#Multiple Linear Regression using LASSO regularization
lasso = Lasso(max_iter=500,alpha = cv_lasso[cv_lasso == min(cv_lasso)].index[0])
lasso.fit(X_train, y_train)
#Predicting the Test set results
lasso_train_lm = lasso.predict(X_train)
lasso_test_lm = lasso.predict(X_test)

#https://www.kaggle.com/fugacity/ridge-lasso-elasticnet-regressions-explained
#Multiple Linear Regression using ELASTICNET regularization
en = ElasticNet(max_iter=500,alpha = cv_elasticnet[cv_elasticnet == min(cv_elasticnet)].index[0])
en.fit(X_train, y_train)
#Predicting the Test set results
en_train_lm = en.predict(X_train)
en_test_lm = en.predict(X_test)


# In[ ]:


#Assessing the Ridge Linear Regression model 
print('Linear Regression Ridge')
print("RMSE on Training set :", rmse_cv(ridge,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(ridge,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, ridge.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, ridge.predict(X_test)))
print('-'*25)

#Assessing the Lasso Linear Regression model 
print('Linear Regression Lasso')
print("RMSE on Training set :", rmse_cv(lasso,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(lasso,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, lasso.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, lasso.predict(X_test)))
print('-'*25)

#Assessing the ElasticNet Linear Regression model 
print('Linear Regression ElasticNet')
print("RMSE on Training set :", rmse_cv(en,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(en,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, en.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, en.predict(X_test)))


# The previous three graphs shows that the range of alpha parameters for the three regularization methods to obtain the best predictions, differ. We show that using the optimal alpha parameter should improve on our prediction. However, there is still a couple more things we can do in order to get the best prediction.

# ### Hyperparameter Cross Validation pipeline

# Theoretically, each alpha parameter should make some predictions better or worse than others. If we take the average of these predictions, we should statistically get a prediction close to the true value. Therefore we use these ranges for the respective methods to make pipelines which constructs models using all of these alpha parameters and combines the predictions from the respective methods in order to obtain the best predictions. 

# In[ ]:


alphas_ridge = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30]
alphas_lasso = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011]
alphas_en = [0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016]
l1ratio_en = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RidgeCV(alphas=alphas_ridge, cv=kfolds))
lasso = make_pipeline(LassoCV(max_iter=500, alphas=alphas_lasso, random_state=42, cv=kfolds))
elasticnet = make_pipeline(ElasticNetCV(max_iter=500,alphas=alphas_en, cv=kfolds, l1_ratio=l1ratio_en))


# In[ ]:


ridge_full = ridge.fit(X_train,y_train)
lasso_full = lasso.fit(X_train,y_train)
en_full = elasticnet.fit(X_train,y_train)


# In[ ]:


#Assessing the Ridge Linear Regression model 
print('Linear Regression Ridge')
print("RMSE on Training set :", rmse_cv(ridge_full,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(ridge_full,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, ridge_full.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, ridge_full.predict(X_test)))
print('-'*25)

#Assessing the Lasso Linear Regression model 
print('Linear Regression Lasso')
print("RMSE on Training set :", rmse_cv(lasso_full,X_train,y_train).mean())
print("RMSE on Test set :", rmse_cv(lasso_full,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, lasso_full.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, lasso_full.predict(X_test)))
print('-'*25)

#Assessing the ElasticNet Linear Regression model 
print('Linear Regression ElasticNet')
print("RMSE on Training set: ", rmse_cv(en_full,X_train,y_train).mean())
print("RMSE on Test set: ", rmse_cv(en_full,X_test,y_test).mean())
print('RMSLE score on train data: ', rmsle(y_train, en_full.predict(X_train)))
print('RMSLE score on test data: ', rmsle(y_test, en_full.predict(X_test)))


# ### Ensembling of models

# Since this did improve our score, what will happen if we use the same concept, but to combine the three different methods? This is called ensembling and works on the same principle. We take a list of models and determine the average predictions between these models. These models can theoretically be weighted to obtain even better predictions, but to keep it simple, we will weigh these models equally.

# In[ ]:


#Assessing the Ensembled model 
print('Ensembled Model')
print('RMSLE score on train data: ', rmsle(y_train, blend_models_predict(X_train,[ridge_full,lasso_full,en_full])))
print('RMSLE score on test data: ', rmsle(y_test, blend_models_predict(X_test,[ridge_full,lasso_full,en_full])))


# Again, we show that there is an improvement by ensembling the models. However, there is still one last thing we can do to improve our model's predictions.

# ### A brute force approach to deal with extreme cases

# From: https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force
# 
# Regression is often bad for edge cases, i.e. for extremely small or big values. Small predictions are overestimating SalePrice, while large predictions are underestimating SalePrice, therefore we scale the predictions at the outer ranges slightly. Note that this only affects a small set of predictions, however it is enough to improve our predictions.
# 
# This following block of code was modified from: From https://www.kaggle.com/agehsbarg/top-10-0-10943-stacking-mice-and-brutal-force

# In[ ]:


blended_train = pd.DataFrame({'Id': X_train.index.values, 'SalePrice': np.expm1(blend_models_predict(X_train,[ridge_full,lasso_full,en_full]))})
blended_test = pd.DataFrame({'Id': X_test.index.values, 'SalePrice': np.expm1(blend_models_predict(X_test,[ridge_full,lasso_full,en_full]))})


# In[ ]:


#We only use the predictions at the extremes, i.e. the 1% quantile and the 99% quantile
q1_tr = blended_train['SalePrice'].quantile(0.01)
q2_tr = blended_train['SalePrice'].quantile(0.99)
q1_te = blended_test['SalePrice'].quantile(0.01)
q2_te = blended_test['SalePrice'].quantile(0.99)

#We scale the identified predictions by lowering the small predictions and increasing the large predictions
blended_train['SalePrice'] = blended_train['SalePrice'].apply(lambda x: x if x > q1_tr else x*0.8)
blended_train['SalePrice'] = blended_train['SalePrice'].apply(lambda x: x if x < q2_tr else x*1.1)
blended_test['SalePrice'] = blended_test['SalePrice'].apply(lambda x: x if x > q1_te else x*0.8)
blended_test['SalePrice'] = blended_test['SalePrice'].apply(lambda x: x if x < q2_te else x*1.1)


# In[ ]:


#Assessing the brute force Ensembled model
print('Ensembled Model')
print('RMSLE score on train data: ', rmsle(y_train, np.log1p(blended_train['SalePrice'])))
print('RMSLE score on test data: ', rmsle(y_test, np.log1p(blended_test['SalePrice'])))


# The brute force approach did improve our RMSLE score on the training set. We will therefore use this strategy to predict the SalePrice for our test data set

# ## Training model for submission
# Now that we know the best way to build our model, we wil train our model on the entire training set and prepare it for submission

# In[ ]:


ridge_full = ridge.fit(train, y)
lasso_full = lasso.fit(train, y)
en_full = elasticnet.fit(train, y)


# In[ ]:


submission = pd.DataFrame({'Id': df_test.index.values, 'SalePrice': np.expm1(blend_models_predict(test,[ridge_full,lasso_full,en_full]))})


# In[ ]:


#We use brute force again to deal with our predictions at the extremes
q1 = submission['SalePrice'].quantile(0.01)
q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.8)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)


# ## Saving prediction for submission
# Here we save our predictions into a file we can submit for scoring

# In[ ]:


submission.to_csv("submission.csv", index=False)
print('Save submission',)


# The model built in this kernel gives us a final score of 0.11749

# In[ ]:




