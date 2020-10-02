#!/usr/bin/env python
# coding: utf-8

# # House prices using advanced regression techniques #
# 
# This is a practical machine learning tutorial based on a kaggle competition named house prices,with 79 explanatory variables describing almost every aspects of residential homes in ames iowa, this competition challenges you to predict the Sale price of each home. 
# 
# ## Goal ##
# It is our job to predict the saleprice of each house in the test set, each row in the test set describes a house and we must predict the sale price of each house.
# 
# ### Metric used for evaluation ###
# 
# Submisssions are evaluated on root-mean squared error(RMSE), between the logarithm of the predicted value and the logarithm of the observed sale price (Taking log, means that errors in predicting expensive houses and cheap houses will affect the result equal.)
# 
# To train the model we have the details of 1460 houses in the train data set. After training , the model will get an idea of what features and qualities of a house decides the sale price. The model will then go through the features of each house in the test set and predict the sale price. 
# We calculate the difference(error) of predicted price from actual price by using a metric called root mean squared error(rmse).

# I used Kaggle course and theese kernels to prepare this solution, thees kernels helped me understand a lot about the basic concepts and many thanks for publishing theese kernels.       
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python    
# https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1    
# https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing   
# 

# Now we are importing necessary packages for this competition.

# In[ ]:



from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import category_encoders as ce
import lightgbm as lgb
from sklearn import metrics
import itertools
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# [](http://)os is the module used to list the directory, we can list the files for this competition from the ../input directory,which is located in the kaggle cloud       
# the sample_submission.csv tells us in which format the predicted saleprice to be submitted. 

# In[ ]:


import os
print(os.listdir("../input"))
print(os.listdir("../input/house-prices-advanced-regression-techniques"))


# We use pandas to read train and test csv files and stores the contents in train and test variables.  

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print ("Data is loaded!")


# ## Exploratory data analysis ##
# 

# Let us examine the data files, train data set has 81 columns including dependent variable the SalePrice, and 1460 rows. 1 row describes details of one house. we later separate and store dependent vriable in y and all 79 independent variables in X one column is id and we will drop that later. 

# In[ ]:


pd.set_option('display.max_columns',1000)
train.head()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


test.head()


# In[ ]:


train.get_dtype_counts()


# In[ ]:


train.describe()


# In[ ]:


train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


# In[ ]:


# Removing all the rows with the target variable is missing.
train.dropna(axis=0, subset=['SalePrice'], inplace=True)
#this will remove the outliers in the column GrLivArea which we will discuss later.
#train = train[train.GrLivArea < 4500]
#train.reset_index(drop=True, inplace=True)

#store target in y and all other indepenent variables in X
y = train.SalePrice
#train.drop(['SalePrice'], axis=1, inplace=True)


# #### Linear regression

# The scatter plot below shows all datapoints around a straight line(This line is caled regression line or bestfit line).The regression line represents the predicted value and the dots represents the actual values.the difference between the actual value and the predicted value is the error, or the distance between each data point and the line is called the error.When the error decreases the predicted value will come close to the actual value.when the error becomes zero then the predicted value will be same as the actual value and the model is considered as 100% accurate, but that is not possible in the real world, because a straight line can never pass through every data points especially when there are thousands of points. But we can find out a best fit line with least sum of squared error. In the coming sections we will explore our data, try to reduce the error and make our model more accurate.

# ![image.png](attachment:image.png)

# ### Assumptons of Liear regression
# 
# Linear regression analyses if one or more independent variable explains the dependent variable.when multiple independent variable involved we use multi linear regresion.
# When we have one target variable and multiple independent variables in the dataset then we have to follow multi linear regression.Target variable must follow a linear relationship with each independent variables for multi linear regression.
# 5 assumptons of linear regression are
# ### 1.   Lenear relationship 
# 
#    Relationship between dependent and independent variables should be linear, and error should be almost same always.
#    
#    
# ### 2.  Homoscedasticity vs Heteroscedasticity
# when the value of the independent variable increases, the error or residual of the target must not increase(It should remain same).It is called  **Homoscedaticity** .if        the residual increases with increase of independent variable is clled **heteroscedasticity**. Target variable should keep homoscedasticity with ecah independent variable,        else it would be a problem for multi linear regression.
#    
#       
# ### 3.   Autocorrelation
# 
# 
# ### 4.   Multivariate normality
# 
#    Target variable must be normally distributed, even multiple independent variables are involved. Transformation of target variable using techniques like log                transformation transform distribution of target variable close to normal distribution. we will visualise this in a moment
#    
#    
# ### 5.   Multicolinearity  
#    Correlation between independent variables are called **Multicollinearity**. multicolinear variables give same information to the model so we      might remove one of them, or use regression models like Lasso or Ridge, theese are good at dealing with multicolinearity 
# 
# 

# 
#  ### Let us check which independent variables are most correlated with target variable.

# In[ ]:


#Find correlation with independent variables and target variable
(train.corr()**2)['SalePrice'].sort_values(ascending = False)[1:]
#This will display correlation with target variable and independent variables from most correlated to least.


# ### Let us check Linear relation ship between target variable and numerical independent variables

# In[ ]:


train1=train.copy()
train.drop(['SalePrice'], axis=1, inplace=True)


# Let us plot numerical independent variables against target variable.

# In[ ]:


def customized_scatterplot(y, x):
        ## Sizing the plot. 
    style.use('fivethirtyeight')
    plt.subplots(figsize = (15,10))
    ## Plotting target variable with predictor variable(OverallQual)
    sns.scatterplot(y = y, x = x);


# In[ ]:


customized_scatterplot(y, train.GrLivArea)


# In[ ]:


#this will remove the outliers in the column GrLivArea.
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)


# Here we can see that price of some houses are too less compared to their Ground living area, we can consider these are outliers.Let us plot each numerical independent variable against target variable and findout some outliers which we will remove from our dataset later. here we will plot few independent variables against 'SalePrice' and findout the rows(Outliers) which we want to remove. we are soon going to findout that whether removing thees rows(Outliers) decreases error(RMSE or MAE) of our model, decreasing error makes our model more accurate.

# In[ ]:


train[train.GrLivArea > 4500]


# In[ ]:


customized_scatterplot(y, train.LotFrontage  )


# In[ ]:


train[train.LotFrontage>300]


# Here from the scatterplot (LotFrontage against SalePrice)we remove row no 934. In all other plots below,if there are more than one outlier, we decided not to delete some rows(even if it appear as an outlier) because it increses the error value, it means while deleting those rows may purging valid information from our dataset.

# In[ ]:


customized_scatterplot(y, train.OpenPorchSF )


# In[ ]:


train[train.OpenPorchSF>500]


# In[ ]:


customized_scatterplot(y, train.LowQualFinSF);


# In[ ]:


train[train.LowQualFinSF>500]


# In[ ]:


customized_scatterplot(y, train.WoodDeckSF)


# In[ ]:


train[train.WoodDeckSF>800]


# In[ ]:


customized_scatterplot(y, train.BsmtFinSF2)


# In[ ]:


train[train.BsmtFinSF2>1400]


# From the above scatter plots it is aparent that SalePrice is not maintaining a good linear relationship with Independent variables. 

# To check each regression assumptions, let us draw a regression line and check how the error of the dependent variable increase when the independent variable increases (Heteroscedatisity).

# In[ ]:


## Plot sizing. 
fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2,sharey=False)
## Scatter plotting for SalePrice and GrLivArea.
sns.scatterplot(x = train.GrLivArea,y = y, ax=ax1)
## regression line for GrLivArea and SalePrice. 
sns.regplot(x=train.GrLivArea, y=y, ax=ax1);

### Scatter plotting for SalePrice and MasVnrArea. 
sns.scatterplot(x = train.MasVnrArea,y = y, ax=ax2)
## regression line for MasVnrArea and SalePrice. 
sns.regplot(x=train.MasVnrArea, y=y, ax=ax2);


# from the above plots it is clear that both masvener area and grlivarea against SalePrice are not maintaining a good linear relationship.

# Below the residualplot will clearly show us the phenmenon heteroscedasticity and linearity.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2,sharey=False)
#plt.subplots(figsize = (15,10))
sns.residplot(train.GrLivArea, y, ax=ax1);
sns.residplot(train.MasVnrArea, y,ax=ax2);


# * Here the datapoints across the straight line is not distributed uniformly around the line. the dots are more widely scattered away from the line or the dots form a cone shape,  The residual or error of the dependent variable increases with respect to the increase in independent variable. 

# Now let us check Multivariate normality. this checks whether our target variable follow a normal distibution. 

# In[ ]:


#y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)


# From theese plots  it is apparent that our target vriable not normally distributed. Johnsonsu is the close match, but even lognormal is also good. Linear regression need target variable to be multivariate normally distributed against independent variables.

# #### Skewness ####
# When the target variable follow a bell shaped curve it is considered normally distributed here the curve is skewed to the right side(curve has tail to the right side) means positive skewness. positive skewness of the data can be normalise to a certain extent with log transformation.

# #### Kurtosis ####
# kurtosis findsout outliers in a dataset. it finds out how long is the tail of the curve is. 

# In[ ]:


print("Skewness: " + str(y.skew()))
print("Kurtosis: " + str(y.kurt()))


# Zero skewness means bell shaped curve not skewed to right or left side. Here the value shows clear positive skewness. Default value of kurtosis is 3, here the value shows so many outliers in the target variable.

# let us apply log transformation on our target variable

# In[ ]:


y = np.log1p(y)
y=y.reset_index(drop=True)


# In[ ]:


#y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2,sharey=False)
#plt.subplots(figsize = (15,10))
sns.residplot(train.GrLivArea, y, ax=ax1);
sns.residplot(train.MasVnrArea, y,ax=ax2);


# After transformation of the target variable, the error variance is reduced significantly this can be seen from the above plots.

# ### Multicolinearity or Correlation between  independent variables ###
# Now it is time to check for multicollenearity it is the correlation between independent variables. Heatmap is best plot to undersrstand multicolinearity. 

# In[ ]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# ### Let us map 10 most correlated variables here. 

# In[ ]:


#saleprice correlation matrix
corrmat = train1.corr()
f, ax = plt.subplots(figsize=(12, 9))
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(train1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# from this heat map it is clear that Garagearea and garage cars are highly corelated.so corellated variables give same nformation to the model so we may remove one of them to eliminate redunduncy, but here we use Lasso or ridge regression both are good at dealing wih multicolinearity.

# #### Missing Values
# This function will help us to count the total no of missing values and its percentage, Machine learning models cannot handle mssing values in feature columns. Following code will give us percentage of missing values in each columns.

# In[ ]:


def missing_percentage(df):
    
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(train)


# Now we will plot the missing data using seaborn bar plot for easy understanding. total 19 columns having missing values with 5 columns having over 50% of mising data. By using several methds we can fill the missing value locations. For example if the values of a particular column is numerical we fill the missing values with mean,median or mode, and with categorical columns fill with most repeated data or use other techniques.

# In[ ]:


sns.set_style("whitegrid")
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[ ]:


#concatenate train and test data 
train_test=pd.concat((train,test)).reset_index(drop=True)


# here we use simple imputer to fill the missing values. simple imputer will decide whether to use mean, median, mode or some other techniques to fill missing data. when we use simple imputer for categorical columns use a strategy like 'most_frequent', for numerical, no strategy is required, imputing is done automatically.  

# In[ ]:


#Separate Numerical and categorical columns.
train_test_num = train_test.select_dtypes(exclude=['object'])
train_test_obj= train_test.select_dtypes(exclude=['int','float'])

my_imputer=SimpleImputer()
my_obj_imputer=SimpleImputer(strategy="most_frequent")
#Impute numerical data
imputed_train_test_num=pd.DataFrame(my_imputer.fit_transform(train_test_num))
#Impute categorical data
imputed_train_test_obj = pd.DataFrame(my_obj_imputer.fit_transform(train_test_obj))
#imputation remove indexes so put column indexes back
imputed_train_test_num.columns = train_test_num.columns
imputed_train_test_obj.columns = train_test_obj.columns
#Concatenate imputed categorical and numerical columns
imputed_train_test=pd.concat([imputed_train_test_obj,imputed_train_test_num],axis=1)


# ### Categorical encoding
# Machine learning models cannot handle categorical values, so that we have to use any one of the encoding technique to convert the categorical columns to numerical columns.Some of the encoding techniques are label encoding, target encoding, count encoding and one hot encoding. Here we use label encoder, count encoder and target ecoder together, and our model produced more accuracy than when onehot encoding used.label encoding will give an unique value to each categorical variables

# ### Count encoding ###
# Each categorical value will be replaced with that value's repeating counts. for example in this dataset feature column Neighborhood, 'OldTown' repeats 113 times, whenever OldTown appears count encoding will replace OldTown with 113.
# ### Target encoding ###
# Target encoding replaces each categorical variable with it's target variable's average value. For example in our train dataset, Neighborhood column's OldTown value will be replaced with average of it's target variable's value. Before doing target encoding we have to split our concatenated data in to train and test set. Target variable is mandatory for target encoding which makes the train test split necessary.

# In[ ]:


pd.set_option('display.max_columns',1000)
imputed_train_test


# In[ ]:


label_encoder = LabelEncoder()
for col in set(train_test_obj):
    imputed_train_test[col]=label_encoder.fit_transform (imputed_train_test[col])


# In[ ]:


pd.set_option('display.max_columns',1000)
imputed_train_test


# In[ ]:


for col in set(train_test_obj): 
    count_enc = ce.CountEncoder(cols=col)
    imputed_train_test[col +'_count']=count_enc.fit_transform(imputed_train_test[col])
    #imputed_train_test[col]=count_enc.fit_transform(imputed_train_test[col])
    


# In[ ]:


pd.set_option('display.max_columns',1000)
imputed_train_test


# In[ ]:


imputed_train_test.shape


# #### Split back concatenated train test data into X_train and X_test

# In[ ]:


X_train = imputed_train_test.iloc[:len(y), :]
X_test = imputed_train_test.iloc[len(y):, :]


# We have to split imputed_train_test dataset into X_train and X_test, because TargetEncoder needs target variable for encoding , and without splitting the dataset X will be 2917 rows and y will be 1460 rows that will through rows mismatch error. 

# In[ ]:


for col in set(train_test_obj):
    target_enc = ce.TargetEncoder(cols=col)
    target_enc.fit(X_train[col],y)
    X_train[col+'_target' ]=target_enc.transform (X_train[col])
    X_test[col+'_target' ]=target_enc.transform(X_test[col])


# In[ ]:


X_train


# In[ ]:


outliers = [30, 88, 462, 631,1322,691,934,297,322,185,53,495]
X_train = X_train.drop(X_train.index[outliers])
y = y.drop(y.index[outliers])

overfit = []
for i in X_train.columns:
    counts = X_train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X_train) * 100 > 99.94:
        overfit.append(i)

overfit = list(overfit)
X_train = X_train.drop(overfit, axis=1)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, train_size=0.8, test_size=0.2,
random_state=0)


# In[ ]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# In[ ]:


ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))


# In[ ]:


gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)


# In[ ]:


lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )


# In[ ]:


xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


# In[ ]:


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)


# In[ ]:


print('START Fit')

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y_train))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(X_train, y_train)

print('Lasso')
lasso_model_full_data = lasso.fit(X_train, y_train)

print('Ridge')
ridge_model_full_data = ridge.fit(X_train, y_train)

print('Svr')
svr_model_full_data = svr.fit(X_train, y_train)

print('GradientBoosting')
gbr_model_full_data = gbr.fit(X_train, y_train)

print('xgboost')
xgb_model_full_data = xgboost.fit(X_train, y_train)

print('lightgbm')
lgb_model_full_data = lightgbm.fit(X_train, y_train)


# In[ ]:


def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) +             (0.05 * lasso_model_full_data.predict(X)) +             (0.1 * ridge_model_full_data.predict(X)) +             (0.1 * svr_model_full_data.predict(X)) +             (0.1 * gbr_model_full_data.predict(X)) +             (0.15 * xgb_model_full_data.predict(X)) +             (0.1 * lgb_model_full_data.predict(X)) +             (0.3 * stack_gen_model.predict(np.array(X))))


# In[ ]:


print('RMSLE score on train data:')
print(rmsle(y_valid, blend_models_predict(X_valid)))


# In[ ]:


print('Predict submission')
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_test)))  


# In[ ]:


#my_model.fit(X_train, y_train)
#preds = my_model.predict(X_test)

#output = pd.DataFrame({'Id': X_test.Id,
                       #'SalePrice': preds})
submission.to_csv('submission.csv', index=False)
submission.to_csv

