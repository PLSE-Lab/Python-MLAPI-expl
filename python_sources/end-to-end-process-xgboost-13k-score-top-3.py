#!/usr/bin/env python
# coding: utf-8

# ## 1.  Introduction
# After finishing the syllabus of the online course and my studies, I'm quite excited about the prospect of starting this project by applying the knowledge and concepts I have picked up. In this project, I will be going the whole end-to-end data science process.
# 
# This project is based on the Kaggle Housing Prices Competition that uses the popular Boston Housing Prices dataset. The dataset includes a list of housing attributes together with the housing price as target variable (label). The purpose is to use machine learning on the dataset to train and develop a model capable of predicting the house price when attributes are inputted. 

# ### 1.1 Dataset
# The following are the features of the dataset.
# 
#  - **SalePrice**  - the property's sale price in dollars. This is the target variable that you're trying to predict.
#  -   **MSSubClass**: The building class
#  -   **MSZoning**: The general zoning classification
#  -   **LotFrontage**: Linear feet of street connected to property
#  -   **LotArea**: Lot size in square feet
#  -   **Street**: Type of road access
#  -   **Alley**: Type of alley access
#  -   **LotShape**: General shape of property
#  -   **LandContour**: Flatness of the property
#  -   **Utilities**: Type of utilities available
#  -   **LotConfig**: Lot configuration
#  -   **LandSlope**: Slope of property
#  -   **Neighborhood**: Physical locations within Ames city limits
#  -   **Condition1**: Proximity to main road or railroad
#  -   **Condition2**: Proximity to main road or railroad (if a second is present)
#  -   **BldgType**: Type of dwelling
#  -   **HouseStyle**: Style of dwelling
#  -   **OverallQual**: Overall material and finish quality
#  -   **OverallCond**: Overall condition rating
#  -   **YearBuilt**: Original construction date
#  -   **YearRemodAdd**: Remodel date
#  -   **RoofStyle**: Type of roof
#  -   **RoofMatl**: Roof material
#  -   **Exterior1st**: Exterior covering on house
#  -   **Exterior2nd**: Exterior covering on house (if more than one material)
#  -   **MasVnrType**: Masonry veneer type
#  -   **MasVnrArea**: Masonry veneer area in square feet
#  -   **ExterQual**: Exterior material quality
#  -   **ExterCond**: Present condition of the material on the exterior
#  -   **Foundation**: Type of foundation
#  -   **BsmtQual**: Height of the basement
#  -   **BsmtCond**: General condition of the basement
#  -   **BsmtExposure**: Walkout or garden level basement walls
#  -   **BsmtFinType1**: Quality of basement finished area
#  -   **BsmtFinSF1**: Type 1 finished square feet
#  -   **BsmtFinType2**: Quality of second finished area (if present)
#  -   **BsmtFinSF2**: Type 2 finished square feet
#  -   **BsmtUnfSF**: Unfinished square feet of basement area
#  -   **TotalBsmtSF**: Total square feet of basement area
#  -   **Heating**: Type of heating
#  -   **HeatingQC**: Heating quality and condition
#  -   **CentralAir**: Central air conditioning
#  -   **Electrical**: Electrical system
#  -   **1stFlrSF**: First Floor square feet
#  -   **2ndFlrSF**: Second floor square feet
#  -   **LowQualFinSF**: Low quality finished square feet (all floors)
#  -   **GrLivArea**: Above grade (ground) living area square feet
#  -   **BsmtFullBath**: Basement full bathrooms
#  -   **BsmtHalfBath**: Basement half bathrooms
#  -   **FullBath**: Full bathrooms above grade
#  -   **HalfBath**: Half baths above grade
#  -   **Bedroom**: Number of bedrooms above basement level
#  -   **Kitchen**: Number of kitchens
#  -   **KitchenQual**: Kitchen quality
#  -   **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms)
#  -   **Functional**: Home functionality rating
#  -   **Fireplaces**: Number of fireplaces
#  -   **FireplaceQu**: Fireplace quality
#  -   **GarageType**: Garage location
#  -   **GarageYrBlt**: Year garage was built
#  -   **GarageFinish**: Interior finish of the garage
#  -   **GarageCars**: Size of garage in car capacity
#  -   **GarageArea**: Size of garage in square feet
#  -   **GarageQual**: Garage quality
#  -   **GarageCond**: Garage condition
#  -   **PavedDrive**: Paved driveway
#  -   **WoodDeckSF**: Wood deck area in square feet
#  -   **OpenPorchSF**: Open porch area in square feet
#  -   **EnclosedPorch**: Enclosed porch area in square feet
#  -   **3SsnPorch**: Three season porch area in square feet
#  -   **ScreenPorch**: Screen porch area in square feet
#  -   **PoolArea**: Pool area in square feet
#  -   **PoolQC**: Pool quality
#  -   **Fence**: Fence quality
#  -   **MiscFeature**: Miscellaneous feature not covered in other categories
#  -   **MiscVal**: $Value of miscellaneous feature
#  -   **MoSold**: Month Sold
#  -   **YrSold**: Year Sold
#  -   **SaleType**: Type of sale
#  -   **SaleCondition**: Condition of sale

# ### 1.2 Importing modules

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)


# ### 1.3 Importing of Data

# In[ ]:


home_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col=0)
home_data


# ## 2. Exploratory Data Analysis

# Lets first look at the characteristics of the data. 

# There are a total of 80 variables including the target variable (Sale Price). There are quite some missing values which will require to be handled later on. There are 37 numerical features and 43 categorical features.

# In[ ]:


print(home_data.info())


# The shape of the data is 1460 rows and 80 columns.

# In[ ]:


home_data.shape


# Below are the descriptive statistics of the data:

# In[ ]:


home_data.describe().round(3)


# ### 2.1 Splitting the data into the target variable
# Below proceding further, lets split the data its features and target variable.

# In[ ]:


target_var_name = 'SalePrice'
target_var = pd.DataFrame(home_data[target_var_name]).set_index(home_data.index)
home_data.drop(target_var_name, axis=1, inplace=True)
target_var


# ### 2.2 Target Variable

# ### 2.2.1 Distribution of target variable (Sale Price)

# The target variable is slightly skewed to the right.

# In[ ]:


print(target_var.describe().round(decimals=2))
sns.distplot(target_var)
plt.title('Distribution of SalePrice')
plt.show()


# ### 2.3 Feature

# ### 2.3.1.1 Numerical feature
# 

# In[ ]:


num_feature = home_data.select_dtypes(exclude=['object']).columns
home_data_num_feature = home_data[num_feature].set_index(home_data.index)


# ### 2.3.1.2 Numerical feature - Univariate analysis

# In[ ]:


home_data_num_feature.describe().round(3)


# In[ ]:


fig = plt.figure(figsize=(12,20))
plt.title('Numerical Feature (before dropping identified outliers)')
for i in range(len(home_data_num_feature.columns)):
    fig.add_subplot(9,4,i+1)
    sns.distplot(home_data_num_feature.iloc[:,i].dropna(),kde_kws={'bw':0.1})
    plt.xlabel(home_data_num_feature.columns[i])

plt.tight_layout()
plt.show()


# ### 2.3.1.3 Numerical Feature - Bivariate analysis
# The scatterplots of SalePrice against each numerical attribute is shown below through the bivariate analysis. 

# In[ ]:


fig = plt.figure(figsize=(12,20))
plt.title('Numerical Feature (before dropping identified outliers)')
for i in range(len(home_data_num_feature.columns)):
    fig.add_subplot(9,4,i+1)
    sns.scatterplot(home_data_num_feature.iloc[:,i], target_var.iloc[:,0])
    plt.xlabel(home_data_num_feature.columns[i])

plt.tight_layout()
plt.show()


# From above, it can be observed that the following points are outliers:
# 
# - LotFrontage (>200)
# - GrLivArea (>4000 AND SalePrice <300000)
# - LowQualFinSF(>550)
# - BsmFinSF1(>4000)
# - LotArea(>100000)
# - 1stFlrSF(>4000)
# - TotalBsmtSF(>6000)
# 
# Outliers will be removed later on.

# ### 2.3.1.4 Correlation among numerical feature
# We shall find the numerical attributes with pearson correlation more than 0.8

# In[ ]:


correlation = home_data_num_feature.corr()

f, ax = plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes', size=16)
sns.heatmap(correlation>0.8)
plt.show()


# The highlightly correlated variables (pearson correlation>0.8) are: 
# - YearBuilt vs GarageYrBlt
# - 1stFlrSF vs TotalBsmtSF
# - GrLivArea vs TotRmsAbvGrd
# - GarageCars vs GarageArea

# ### 2.3.1.5 Correlation between numerical features and target variable

# In[ ]:


y_corr = pd.DataFrame(home_data_num_feature.corrwith(target_var.SalePrice),columns=["Correlation with target variable"])
# plt.hist(y_corr)


# In[ ]:


y_corr_sorted= y_corr.sort_values(by=['Correlation with target variable'],ascending=False)
y_corr_sorted


# In[ ]:


fig = plt.figure(figsize=(6,10))
plt.title('Correlation with target variable')
a=sns.barplot(y_corr_sorted.index,y_corr_sorted.iloc[:,0],data=y_corr)
a.set_xticklabels(labels=y_corr_sorted.index,rotation=90)
plt.tight_layout()
plt.show()


# We will have to remove each of the pair that are highly correlated when we are cleaning the data later. One feature can be represented by the other, so there is no need for both. We will keep the feature of the pair that has more correlation with the target variable and remove the feature with the lesser correlation. The features that will be removed is in bold.
# - YearBuilt vs **GarageYrBlt**
# - **1stFlrSF** vs TotalBsmtSF
# - GrLivArea vs **TotRmsAbvGrd**
# - GarageCars vs **GarageArea**
# 

# Features that have very weak correlation with the target variable should also be removed (-0.1 < pearson correlation < 0.1).

# In[ ]:


[(y_corr_sorted<0.1) & (y_corr_sorted>-0.1)]


# Features to be removed due to low correlation with target variable:
#  - PoolArea                                   
#  - MoSold                                     
#  - 3SsnPorch                                  
#  - BsmtFinSF2                                 
#  - BsmtHalfBath                               
#  - MiscVal                                    
#  - LowQualFinSF                               
#  - YrSold                                     
#  - OverallCond                                
#  - MSSubClass                     
# 

# ### 2.3.2 Categorical Feature

# In[ ]:


cat_feature = home_data.select_dtypes(include=['object']).columns
home_data_cat_feature = home_data[cat_feature]


# ### 2.3.2.1 Distribution of Categorical Feature

# In[ ]:


fig = plt.figure(figsize=(18,50))
plt.title('Distribution of Categorical Feature')
for i in range(len(home_data_cat_feature.columns)):
    fig.add_subplot(15,3,i+1)
    sns.countplot(home_data_cat_feature.iloc[:,i])
    plt.xlabel(home_data_cat_feature.columns[i])
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# ### 2.3.2.2 Boxplot of Categorical Feature

# In[ ]:


fig = plt.figure(figsize=(18,80))
plt.title('Numerical Feature (before dropping identified outliers)')
for i in range(len(home_data_cat_feature.columns)):
    fig.add_subplot(15,3,i+1)
    sns.boxplot(x=home_data_cat_feature.iloc[:,i],y=target_var['SalePrice'])
    plt.xlabel(home_data_cat_feature.columns[i])
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# In[ ]:


##look into sorting by median


# ## 3. Data Cleaning and Preprocessing

# ### 3.1 Dropping unnecessary features

# In[ ]:


home_data_num_feature=home_data_num_feature.drop(['GarageYrBlt','1stFlrSF','TotRmsAbvGrd','GarageArea'],axis=1)
home_data_num_feature.columns


# In[ ]:


home_data_num_feature=home_data_num_feature.drop(['PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass'],axis=1)


# In[ ]:





# ### 3.2 Removing Outliers
# We shall now drop the outliers.

# In[ ]:


home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['LotFrontage']>300].index)
print(len(home_data_num_feature))
home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[(home_data_num_feature['GrLivArea']>4000) & (target_var['SalePrice']<300000)].index)
print(len(home_data_num_feature))

home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['BsmtFinSF1']>4000].index)
print(len(home_data_num_feature))

home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['LotArea']>100000].index)
print(len(home_data_num_feature))

home_data_num_feature=home_data_num_feature.drop(home_data_num_feature[home_data_num_feature['TotalBsmtSF']>6000].index)
print(len(home_data_num_feature))


# In[ ]:


fig = plt.figure(figsize=(12,12))
plt.title('Numerical Feature (after dropping identified outliers)')
for i in range(len(home_data_num_feature.columns)):
    fig.add_subplot(6,4,i+1)
    sns.scatterplot(home_data_num_feature.iloc[:,i], target_var.iloc[:,0])
    plt.xlabel(home_data_num_feature.columns[i])

plt.tight_layout()
plt.show()


# ### 3.3 Handling Missing Data - Numerical Feature

# We will first manage the missing numerical feature data. Only LotFontage has some missing data. I do not think it is a big issue as only a small proportion of data is missing.

# In[ ]:


home_data_num_feature.count()


# We will use SimpleImputer to fill the missing data. SimpleImputer will fill the missing values with the mean value for that feature.

# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer()
home_data_num_feature = pd.DataFrame(imp.fit_transform(home_data_num_feature),columns=home_data_num_feature.columns,index=home_data_num_feature.index)
home_data_num_feature.count()


# ### 3.4 Handling Missing Data - Categorical Feature

# For categorical data, there are some features with a high proportion of missing data. This will be an issue.

# In[ ]:


home_data_cat_feature.count()


# We should remove Alley, PoolQc, Fence and Misc Feature. The proportion of non-missing values to the sample size is too small. 
# For these feature, Uni-variate impuding doesnt make sense as that will mean adding the mean of the feature as information.
# Multi-variate impuding may make sense as we can draw relationships the feature have with others to estimate their missing values. However, multi-variate impuding for catergorical data is not supported yet in libraries. 
# 
# We shall drop these 4 features. 

# In[ ]:


home_data_cat_feature=home_data_cat_feature.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1)


# For the remaining categorical feature, we will impute the missing values with "most frequent" strategy. 

# In[ ]:


imp = SimpleImputer(strategy="most_frequent")
home_data_cat_feature=pd.DataFrame(imp.fit_transform(home_data_cat_feature),columns=home_data_cat_feature.columns,index=home_data_cat_feature.index)
home_data_cat_feature


# ### 3.5 Encoding Categorical Feature

# We will use OneHotEncoders instead of drop_dummies because ...

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(drop='first',sparse=False)
enc.fit(home_data_cat_feature)
home_data_cat_feature_dummies=enc.transform(home_data_cat_feature)
home_data_cat_feature_dummies = pd.DataFrame(home_data_cat_feature_dummies,columns=enc.get_feature_names(),index=home_data_cat_feature.index)
home_data_cat_feature_dummies


# ### 3.6 Combining Feature

# In[ ]:


X=pd.merge(home_data_num_feature,home_data_cat_feature_dummies,how='left',left_index=True,right_index =True)

# X=pd.concat([home_data_num_feature,home_data_cat_feature_dummies],axis=1)
y=target_var.loc[X.index]


# In[ ]:


X


# ## 4 Modelling

# We will be using K-fold Cross Validation as the validation method. The training data set (X and y) that we have processed so far will be split in 5 fold. One of the fold will be used as the validation set while the rest will be used to training the data. The data will be trained using the following algorithms:
# 
# - Linear Regression
# - Linear Regression (Lasso)
# - Linear Regression (Ridge)
# - Decision Tree
# - Decision Tree with Bagging
# - RandomForestRegressor
# - Adaboost regressor
# - GradientBoosting
# - XGBoost
# - Support Vector Regressor
# 
# During training, Gridsearch will be used to search for the parameters. As Gridsearch will run all the parameters to determine the best parameters, it could lead to over-fitting of the training set. Over-fitting means that the model is fitted too closely and specifically to the training set. However, when making prediction on new data set, it will not be accurate as the model could not be generalised beyond the training set. To prevent overfitting, Gridsearch will determine the best parameter by the best score (lowest RMSE) on the validation set.'
# 
# We will then use the models above and their gridsearch optimized parameters to make prediction of the test set. The predictions will be uploaded to Kaggle website to determine the model's performance. 
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

r2 = make_scorer(r2_score)
rmse = make_scorer(mean_squared_error,greater_is_better=False,squared=False)

cv_list={}
cv_rmse={}
cv_r2={}
cv_best_mse={}


# ### 4.1 Linear Regression

# In[ ]:


model_name = "LinearRegression"
model=LinearRegression()

param_grid = [{model_name+'__fit_intercept':[True,False]}]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.2 Lasso

# In[ ]:


from sklearn.linear_model import Lasso

model_name = "Lasso"
model=Lasso()

param_grid = [  {model_name+'__'+'alpha': [2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.3 Ridge

# In[ ]:


from sklearn.linear_model import Ridge

model_name = "Ridge"
model=Ridge()

param_grid = [{model_name+'__'+'alpha': [2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.3 Decision Trees

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

model_name='DecisionTreeRegressor'
model=DecisionTreeRegressor()

param_grid = [{model_name+'__'+'splitter': ['best','random'],
              model_name+'__'+'max_depth':np.arange(1,20)
              }]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.4 Decision Tree with Bagging

# In[ ]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

model_name = 'BaggingDecisionTreeRegressor'
model=BaggingRegressor(DecisionTreeRegressor())

param_grid = [{model_name+'__'+'base_estimator__splitter': ['best','random'],
              model_name+'__'+'base_estimator__max_depth':np.arange(1,30)
              }]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy().ravel())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.5 RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model_name='RandomForestRegressor'
model=RandomForestRegressor()

param_grid = [{model_name+'__'+'max_depth' : np.arange(1,100,2)}]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy().ravel())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.6 AdaBoostRegressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

model_name='AdaBoostRegressor'
model=AdaBoostRegressor()

param_grid = [{model_name+'__'+'learning_rate' : [0.001,0.01,0.1,1,10,100]}]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy().ravel())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.7 GradientBoosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

model_name='GradientBoostingRegressor'
model=GradientBoostingRegressor()

param_grid = [{model_name+'__'+'loss' : ['ls','lad','huber','quantile'],model_name+'__'+'learning_rate' : [0.01,0.1,1,10],model_name+'__'+'criterion':['friedman_mse', 'mse']}]


# In[ ]:


pipeline = Pipeline([(model_name, model)])


reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
reg.fit(X,y.to_numpy().ravel())


#Record the best grid search paramters into the list.
cv_list[model_name]=reg
cv_rmse[model_name]=reg.best_score_

#print out the best param and best score
print(model_name)
print('best training param:',reg.best_params_)
print('best training score rmse', reg.best_score_)
print('\n')


# ### 4.8 XGBoost

# In[ ]:


from xgboost import XGBRegressor
model_name='XGBoost'
model=XGBRegressor()

param_grid = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [3,4,5],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}


# In[ ]:



xgb_dt=GridSearchCV(model, param_grid,n_jobs=-1,cv=5,scoring=rmse)
xgb_dt.fit(X,y)

cv_list[model_name]=xgb_dt.best_estimator_
cv_rmse[model_name]=xgb_dt.best_score_

print(xgb_dt.best_estimator_)
print(xgb_dt.best_score_)


# ### 4.9 Support Vector Machine

# In[ ]:


# from sklearn.svm import SVR

# model_name = "SVR"
# model=SVR()

# param_grid = [
#   {model_name+'__'+'C': [0.1,1], model_name+'__'+'kernel': ['linear','poly','rbf','sigmoid'],
#    model_name+'__'+'gamma':['auto']
#   }]


# In[ ]:


# pipeline = Pipeline([(model_name, model)])

# reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=2)
# reg.fit(X,y.to_numpy().ravel())

# #Record the best grid search paramters into the list.
# cv_list[model_name]=reg
# cv_rmse[model_name]=reg.best_score_

# #print out the best param and best score
# print(model_name)
# print('best training param:',reg.best_params_)
# print('best training score rmse', reg.best_score_)
# print('\n')


# ### 4.10 Summary of Cross Validation results

# In[ ]:


score = abs(pd.DataFrame.from_dict(cv_rmse,orient='index',columns=['CV Score']))
score = score.sort_values('CV Score')
score


# We observed that XGBoost is the best performer with decision tree regressor the worst. In this project, as mentioned above, we shall make predictions on all models and send all predictions for submission to get the score. 

# ## 5. Prediction

# We first import test data set.

# In[ ]:


test_data_path = '../input/house-prices-advanced-regression-techniques/test.csv'
X_test_set = pd.read_csv(test_data_path,index_col=0)
X_test_set.shape


# Next, we manage the numerical feature of the dataset

# In[ ]:


test_home_data_num_feature = X_test_set[home_data_num_feature.columns]
test_home_data_num_feature.describe().round(2)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
test_home_data_num_feature = pd.DataFrame(imp.fit_transform(test_home_data_num_feature),columns=test_home_data_num_feature.columns,index=test_home_data_num_feature.index)
# test_norm_home_data_num_feature = pd.DataFrame(scaler.fit_transform(test_home_data_num_feature),columns=test_home_data_num_feature.columns,index=test_home_data_num_feature.index)
test_home_data_num_feature


# Then, we handle the catergorical feature of the dataset.

# In[ ]:


test_home_data_cat_feature = X_test_set[home_data_cat_feature.columns]
test_home_data_cat_feature.describe().round(2)

imp = SimpleImputer(strategy="most_frequent")
test_home_data_cat_feature=pd.DataFrame(imp.fit_transform(test_home_data_cat_feature),columns=test_home_data_cat_feature.columns,index=test_home_data_cat_feature.index)
test_home_data_cat_feature

test_home_data_cat_feature_dummies = enc.transform(test_home_data_cat_feature)

test_home_data_cat_feature_dummies = pd.DataFrame(test_home_data_cat_feature_dummies,columns=enc.get_feature_names(),index=test_home_data_cat_feature.index)
test_home_data_cat_feature_dummies


# Merging both numerical features and categorical features of the test set

# In[ ]:


X_test = pd.concat([test_home_data_num_feature,test_home_data_cat_feature_dummies],axis=1)
X_test


# ### 5.1 Predicting target variable
# We will now predict the target variable with the processed test set features.

# In[ ]:


predict=cv_list['LinearRegression'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict[:,0]},index=X_test.index)
output.to_csv('LinearRegression.csv', index=True)


# In[ ]:


predict=cv_list['Lasso'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict},index=X_test.index)
output.to_csv('Lasso.csv', index=True)


# In[ ]:


predict=cv_list['Ridge'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict[:,0]},index=X_test.index)
output.to_csv('Ridge.csv', index=True)


# In[ ]:


predict=cv_list['DecisionTreeRegressor'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict},index=X_test.index)
output.to_csv('DecisionTreeRegressor.csv', index=True)


# In[ ]:


predict=cv_list['BaggingDecisionTreeRegressor'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict},index=X_test.index)
output.to_csv('BaggingDecisionTreeRegressor.csv', index=True)


# In[ ]:


predict=cv_list['RandomForestRegressor'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict},index=X_test.index)
output.to_csv('RandomForestRegressor.csv', index=True)


# In[ ]:


predict=cv_list['AdaBoostRegressor'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict},index=X_test.index)
output.to_csv('AdaBoostRegressor.csv', index=True)


# In[ ]:


predict=cv_list['GradientBoostingRegressor'].predict(X_test)
output = pd.DataFrame({'SalePrice': predict},index=X_test.index)
output.to_csv('GradientBoostingRegressor.csv', index=True)


# In[ ]:


predict=xgb_dt.predict(X_test)
output = pd.DataFrame({'SalePrice': predict},index=X_test.index)
output.to_csv('xgb.csv', index=True)


# ## 6. Submission for scoring and Summary of results

# All excel output files are submitted to Kaggle to be checked against their hidden test set.
# 
# The following are the results:

# In[ ]:


test_score = {'LinearRegression':17183.86239,'Lasso':16379.19466,'Ridge':16107.36134,'DecisionTreeRegressor':24232.59348,'BaggingDecisionTreeRegressor':17949.15006,'RandomForestRegressor':16163.46606,'AdaBoostRegressor':22434.87007,'GradientBoostingRegressor':15517.90164,'XGBoost':13745.37874}


# In[ ]:


test_score = pd.DataFrame.from_dict(test_score,orient='index')
score['Test Score']=test_score
score


# In this project, XGBoost is the best performing model with a score of 13745. This achieved ranking of 891 (top 3%) on the leadership board.

# ## 7. Future Studies

# My personal thoughts are that more could be done for the top 4 regression model. 
# 
# ### 1. Principal Component Analysis
# Principal Component Analysis could be implement on the features. This could be key in improving the score given the amount of dummy variable we have (196). In the project above, the dummy variables couldnt be processed. Unlike the numerical variable which correlation can be studied and the correlated features be dropped, we are unable to do so the dummy variables. Principal component analysis provide a way to reduce the large amount of dummy variables to an optimal size. 

# ### 2. Standardisation
# Standardisation should improve the score for the non-tree based regression methods. We can apply standardisation to see whether the score improve for Ridge and Lasso.

# ### 3. Multi-variate Imputation
# For the categorical feature, we can look into how to use the relationships that the feature of the missing values shared with other features to derive the missing values. This perhaps can help us to make estimation for features with large proportion of missing values. So far, we have dropped quite some catergorical features because of the missing values and our limitation to the use of uni-variate imputation. There may be some difficulty on this as multi-variate imputation for categorical feature is not supported by sklearn yet.

# ### 4. Ensemble
# Ensemble can used on all models to see if the aggregated results from hard-voting or soft-voting yield better score. This should be easy to implement.

# In[ ]:




