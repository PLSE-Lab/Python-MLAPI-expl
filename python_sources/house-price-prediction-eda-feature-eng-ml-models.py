#!/usr/bin/env python
# coding: utf-8

# This is my first kaggle competition and a stepping stone for my career in the field of Data Science!                      
# I have taken inspiration from [Stacked Regressions to predict House prices](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook). 
# 

# ### Exploratory Data Analysis

# The given training data contains 1460 rows and 81 columns. The first thing is to get the data in the right format before diving into predicting the models. My analysis involves the following steps:
# 1. Understanding the response/dependent variable - `SalePrice`
# 2. Understanding the relation between the dependent varaible and independent variables.
# 3. Cleaning the data - handling missing values, oultiers and categorical variables.

# #### Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Supress the warnings that we get from sklearn and seaborn

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# #### Importing the data from the .csv file

# In[ ]:


traindata = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
testdata = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# #### Checking out the features present in the data

# In[ ]:


traindata.info()


# **Exploring outliers!**

# While analysing the data set, I found that there are quite a few outliers. Since, we have very less data (around 1460 rows), removing outliers with respect to all the features will reduce the data for our analysis. I am considering 'Ground Living Area' predictors outliers. So, just removing a few extremely affecting outliers is safe because removing all of them may greatly affect the models if there were also outliers in the test data. 

# In[ ]:


#Before dropping the outliers
fig, ax = plt.subplots()
ax.scatter(x = traindata['GrLivArea'], y = traindata['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.title("Before dropping the outliers")
plt.show()

#Deleting outliers
traindata = traindata.drop(traindata[(traindata['GrLivArea']>4000) & (traindata['SalePrice']<300000)].index)

#After dropping the outliers
fig, ax = plt.subplots()
ax.scatter(traindata['GrLivArea'], traindata['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.title("After dropping the outliers")
plt.show()


# There are two outliers in the first subplot. They have an unconvincing low price but with larger GrLivArea and they do not follow the trend of the data, hence we can safely remove them.                    
# Also, there are two more extreme point over $600000 which are not outliers, they are following the trend of the data, hence we retain them.

# #### Understanding the statistics of the response/dependent variable - `'SalePrice'`

# In[ ]:


traindata['SalePrice'].describe()


# #### Looking at how the variable is distributed, by plotting a histogram and QQ-Plot

# In[ ]:


#Histogram
sns.distplot(traindata['SalePrice'])

#QQ-Plot
fig = plt.figure()
res = stats.probplot(traindata['SalePrice'], plot=plt)


# From the above histogram, 'SalePrice' is deviated and right skewed from normal distribution. According to the Central Limit Theorem, all the RV's tends to be normally distributed and hence we require the response variable to be transformed into a normally distributed variable.                           
# QQ-plot also shows that the response variable is not normally distributed where the data does not follow along the line that is normally distributed. 

# In[ ]:


print("Skewness before transformation:", round(traindata['SalePrice'].skew(),5))
print("Kurtosis before transformation:", round(traindata['SalePrice'].kurt(),5))


# #### Transforming 'SalePrice' to make it normally distributed.
# 
# Usually, log transforms help.

# In[ ]:


traindata['SalePrice'] = np.log(traindata['SalePrice'])


# #### Plotting the transformed variable - Now, it should be normally distributed

# In[ ]:


#Histogram
sns.distplot(traindata['SalePrice'])

#QQ-Plote
fig = plt.figure()
res = stats.probplot(traindata['SalePrice'], plot=plt)


# In[ ]:


print("Skewness after transformation:", round(traindata['SalePrice'].skew(),5))
print("Kurtosis after transformation:", round(traindata['SalePrice'].kurt(),5))


# Skewness and kurtosis have significantly reduced.

# #### Feature Engineering

# Concatenating the train and test data to a single dataframe - This step helps us to impute the missing values.

# In[ ]:


ntrain = len(traindata)
y = traindata['SalePrice']

combined_df = pd.concat([traindata,testdata], ignore_index=True)
combined_df.drop(columns= 'SalePrice', inplace = True)
print("Shape of combined data frame: ", combined_df.shape)


# #### Handling Missing values

# Calculating the missing ratio to see which features have a greater percentage of missing data.

# In[ ]:


#This heatmap shows the missing data present in the data frame
plt.figure(figsize=(20,3))
sns.heatmap(combined_df.isnull(), yticklabels=False, cbar=False, cmap = 'viridis')


# In[ ]:


(combined_df.isnull().sum().sort_values(ascending = False) / len(combined_df)) * 100


# #### Imputing the missing values
# 
# We see that **`PoolQC`**, **`MiscFeature`**, **`Alley`**, **`Fence`**, **`FireplaceQU`** have a high ratio of missing values (also seen in the map). But, according to the data description, NA in all of these features means that there is no such feature in the house. For example, NA in PoolQC means that there is no pool in the house, NA in alley means that there is no alley access etc. Therefore, we impute these values by filling **None**.

# In[ ]:


combined_df['PoolQC'] = combined_df['PoolQC'].fillna('None')
combined_df['MiscFeature'] = combined_df['MiscFeature'].fillna('None')
combined_df['Alley'] = combined_df['Alley'].fillna('None')
combined_df['Fence'] = combined_df['Fence'].fillna('None')
combined_df['FireplaceQu'] = combined_df['FireplaceQu'].fillna('None')


# **`LotFrontage`** is the linear feet of street connected to the property. This means that the area of street connected to each house will mostly be similar and hence houses in the neighbourhood also tend to have the same area of street. We can impute the missing values in this feature by **median LotFrontage** of the neighbourhood.

# In[ ]:


combined_df['LotFrontage'] = combined_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# For all other features with less missing ratio, we also impute the missing values by filling it with **None**.

# In[ ]:


for columns in ('GarageType','GarageFinish', 'GarageQual','GarageCond','MSSubClass','BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType'):
    combined_df[columns] = combined_df[columns].fillna('None')


# For features that are quantitative, we fill the missing values with **0**. For example, missing values in GarageCars means there are no cars in the garage and hence we impute it with 0.

# In[ ]:


for columns in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    combined_df[columns] = combined_df[columns].fillna(0)


# For feautures **`MSZoning`**, **`Electrical`**, **`KitchenQual`**, **`Exterior1st`**, **`Exterior2nd`**, **`SaleType`**, we fill the missing values with the **mode** of the predictors as these features have values that occur most frequently. 

# In[ ]:


for columns in ('MSZoning','Electrical', 'KitchenQual','Exterior1st','Exterior2nd','SaleType'):
    combined_df[columns] =  combined_df[columns].fillna(combined_df[columns]).mode()[0]


# In **`Functional`** feature, we have 2 NA's. NA means 'Typical' according to the data description and hence we fill them with **Typ** and we drop the **`Utilities`** feature as it has all its values as 'AllPub' and is of no use to us.

# In[ ]:


combined_df['Functional'] =  combined_df['Functional'].fillna('Typ')
combined_df.drop(columns='Utilities', axis = 1, inplace = True)


# Checking to see if there are missing values in any other features.

# In[ ]:


(combined_df.isnull().sum().sort_values(ascending = False) / len(combined_df)) * 100


# Missing ratio is 0 for all the features. Hence, we can confirm that we do not have missing values in the data.

# #### A little more feature engineering

# We notice that two of the numerical features are actually categorical. Let's transform them to categorical type.

# In[ ]:


combined_df['MSSubClass'] =  combined_df['MSSubClass'].apply(str)
combined_df['OverallCond'] = combined_df['OverallCond'].astype(str)


# Any feature which is related to 'area' is important in determing the house prices. We add a new feature for obtaining the 'total area' using TotalBsmtSF, 1stFlrSF, 2ndFlrSF. 

# In[ ]:


combined_df['TotalSF'] = combined_df['TotalBsmtSF'] + combined_df['1stFlrSF'] + combined_df['2ndFlrSF']


# #### Skewed Features
# 
# After exploring the skewness in our response variable, now, exploring the skewness in the numerical features.

# In[ ]:


#Getting only the numerical features
numeric_features = combined_df.dtypes[combined_df.dtypes != 'object'].index

#Checking the skewness of all the numerical features
skewed_features = combined_df[numeric_features].apply(lambda x: x.skew()).sort_values(ascending = False)
skewed_df = pd.DataFrame({'Skew': skewed_features})
display(skewed_df)


# Transforming the highly skewed features to make it Gaussian using Box-Cox transformation. 

# In[ ]:


from scipy.special import boxcox1p

skewed_df = skewed_df[abs(skewed_df) > 0.75]
skewed_feats = skewed_df.index
lam = 0.15
for feats in skewed_feats:
    combined_df[feats] = boxcox1p(combined_df[feats], lam)


# #### Categorical Variables

# Let's assign dummy variables to categorical features for modelling purposes(eg: Linear Regression requires the creation of dummy variables for modelling)

# In[ ]:


combined_df = pd.get_dummies(combined_df, drop_first=True)
combined_df.shape


# Splitting the concatenated data back to train and test set to start our modelling.

# In[ ]:


X_train = combined_df[:traindata.shape[0]]
X_test = combined_df[traindata.shape[0]:]
y = traindata.SalePrice


# All our predictors are in 'X_train' and response is in 'y'.

# ### Modelling!

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse.mean())


# #### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

lm_rmse = rmse_cv(LinearRegression())
print("RMSE for Linear Regression: ", lm_rmse)


# #### LASSO
# 
# LASSO requires to know the value of alpha. Alpha can be obtained through cross validation 

# In[ ]:


from sklearn.linear_model import Lasso, LassoCV

lassocv = LassoCV(cv=5,random_state=1)
lassocv.fit(X_train, y)
best_alpha = lassocv.alpha_

lasso_model = make_pipeline(RobustScaler(), Lasso(alpha= best_alpha, random_state=1))
lasso_rmse = rmse_cv(lasso_model)
print("RMSE for LASSO (L1 Regularization): ", lasso_rmse)


# #### Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV

ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas=np.logspace(-10,10,100)))
ridge_rmse = rmse_cv(ridge_model)
print("RMSE for Ridge Regression (L2 Regularization): ", ridge_rmse)


# #### Elastic Net : Hybrid of LASSO and Ridge Regression

# In[ ]:


from sklearn.linear_model import ElasticNet, ElasticNetCV

elasticnet_cv = ElasticNetCV(l1_ratio=np.arange(0.1,1,0.1), cv=5, random_state=1)
elasticnet_cv.fit(X_train, y)
best_l1_ratio = elasticnet_cv.l1_ratio_
best_aplha = elasticnet_cv.alpha_

elasticnet_model = make_pipeline(RobustScaler(), ElasticNet(alpha=best_alpha, l1_ratio= best_l1_ratio, random_state=1))
elasticnet_rmse = rmse_cv(elasticnet_model)
print("RMSE for Elastic Net(L1 and L2 Regularization): ", elasticnet_rmse)


# #### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

randomforest_model = rmse_cv(RandomForestRegressor(random_state=1))
print("RMSE for Random Forest: ", randomforest_model)


# #### Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

gradientboost_model = GradientBoostingRegressor(learning_rate=0.1, loss='huber', n_estimators=3000, random_state=1)
gradientboost_cv = rmse_cv(gradientboost_model)
print("RMSE for Gradient Boost: ", gradientboost_cv)


# #### Adaptive Boosting with LASSO as the estimator

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

adaboost_lassomodel = AdaBoostRegressor(lasso_model, n_estimators=50, learning_rate=0.001, random_state=1)
adaboost_lassocv = rmse_cv(adaboost_lassomodel)
print("RMSE for Adaptive Boosting with LASSO estimator: ", adaboost_lassocv)


# #### Adaptive Boosting with Elastic Net as the estimator

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

adaboost_enetmodel = AdaBoostRegressor(elasticnet_model, n_estimators=50, learning_rate=0.001, random_state=1)
adaboost_enetcv = rmse_cv(adaboost_enetmodel)
print("RMSE for Adaptive Boosting with Elastic Net estimator: ", adaboost_enetcv)


# After applying all the models to our data, Adaptive Boosting with Elastic net as the estimator seems to give a better rmse when compared to all other models. 

# **Principal Component Analysis**
# 
# Trying to reduce the dimensionality of the data to see if it helps in our predictions.

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

pca = make_pipeline(RobustScaler(), PCA(n_components = 3, random_state = 1))
pca.fit(X_train.transpose())
print(f"Proportion of variance explained by the components: {pca.steps[1][1].explained_variance_ratio_}")

# we are using 3 components in this case
p_comps = pca.steps[1][1].components_.transpose()

pca_lm_rmse = np.sqrt(-cross_val_score(LinearRegression(), p_comps, y, cv = 5, scoring = "neg_mean_squared_error")).mean()
print(f"RMSE for Linear Regression after PCA reduction: [{pca_lm_rmse}]")


# PCA also doesn't help because the RMSE has increased after the PCA reduction when compared to before reduction. So, we choose Adaptive boosting with Elastic Net as the estimator and predict the prices.

# **Prediction time!**

# In[ ]:


adaboost_enetmodel.fit(X_train, np.exp(y))
submission_predictions = adaboost_enetmodel.predict(X_test)


# ### Submission

# In[ ]:


results = pd.DataFrame({'Id': testdata['Id'], 'SalePrice':submission_predictions})
results.to_csv("submission.csv", index = False)

