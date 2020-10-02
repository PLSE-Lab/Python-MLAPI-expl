#!/usr/bin/env python
# coding: utf-8

# # Problem Statment
# 
# Build a model to solve a problem of predicting house prices with giving the minimum RMSE score prediction on prices based on diffrent features. This program will help the house sellers and also the buyers to get the model predict the located house price based on it current features
# 

# # Introduction

# The notebook is to create an accurate prediction model for house pricing  based on historical data that has +80 features.
# 
# The features are classified to numerical and categorical data. Our task is to study the correlation between each feature, clean the data by replacing nulls, dropping unnecessary features, and converting data type for some features. After cleaning part, we will standarize the distribution of feature values to have scaled model. Then we will select features of our model and apply different regression approaches to decide what is the most accurate model.
# 
# The consequential steps to have our model are as the following:
#  1. Import libraries and datasets
#  2. Exploratory Data Analysis<br>
#       2.1 Data Overview<br>
#       2.2 Statistics<br>
#       2.3 Correlation analysis<br>
#  3. Data Cleaning<br>
#       3.1 Training dataframe cleaning<br>
#       3.2 Finding the outliers<br>
#       3.3 Teasting dataframe cleaning
#  4. Get Dummies<br>
#       4.1 find categorial features
#  5. Select Features
#  6. Apply different Prediction models<br>
#     6.1 LassoCV model with standarize<br>
#     6.2 RidgeCV model<br>
#         6.2.1 RidgeCV modelwithout standarize<br>
#         6.2.2 RidgeCV model with standarize<br>
#     6.3 Bagging regresser<br>
#         6.3.1 Bagging regrosser without standarize<br>
#         6.3.2 Bagging regrosser with standarize<br>
#     6.4 Random forest regresser<br>
#         6.4.1 Random forest regresser without standarize<br>
#         6.4.2 Random forest regresser with standarize<br>
#     6.5 Decision Tree regressor with standarize<br>
#         6.5.1 Decision Tree regressor with standarize<br>
#         6.5.2 Decision Tree regressor without standarize<br>
#     6.6 models with Feature selection<br>  
#  7. Sumbission 

# In[ ]:


import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LassoCV,LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict,train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm, skew 
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.filterwarnings("ignore")


# # 1. Import Training and Testing Dataset

# In[ ]:


house_prices_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_prices_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sampl_sub=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# # 2. Exploratory Data Analysis

# ## 2.1 Data Overview

# In[ ]:


house_prices_train.head()


# In[ ]:


house_prices_test.head()


# In[ ]:


house_prices_train.shape


# In[ ]:


house_prices_test.shape


# In[ ]:


sampl_sub.shape


# In[ ]:


house_prices_train.info()


# ## 2.2 Statistics

# In[ ]:


house_prices_train.describe()


# In[ ]:


#check the column type 
house_prices_train.dtypes


# In[ ]:


#change the column type to it's appropriate type if possible
house_prices_train=house_prices_train.apply(pd.to_numeric, errors='ignore')


# In[ ]:


#check the numbers of uniques values in all columns 
house_prices_train.nunique(axis=0)


# In[ ]:


#ckeck if there is a null in the train dataframe

print("is there missing samples in training set:",house_prices_train.isnull().values.any())

# get all the columns with a missing values
house_prices_train.columns[house_prices_train.isnull().any()]


# In[ ]:


#ckeck if there is a null in the test dataframe


print("is there missing samples in test set:",house_prices_test.isnull().values.any())

# get all the columns with a missing values
house_prices_train.columns[house_prices_train.isnull().any()]


# In[ ]:


# Check the statistical measures of the predicted target (SalePrice)
house_prices_train['SalePrice'].describe()


# In[ ]:


#check SalePrice distribuation
plt.figure(figsize=(10,5));
plt.xlabel('xlabel', fontsize=16);
plt.rc('xtick', labelsize=14); 
plt.rc('ytick', labelsize=14); 


sns.distplot(house_prices_train['SalePrice']);
print("Skewness: %f" % house_prices_train['SalePrice'].skew())


# The SalePrices is Positive skew this is mean there are outliers greater than the mean.

# In[ ]:


# YrSold has the lowest correlation with SalePrice
sns.boxplot(x="YrSold",y="SalePrice",data=house_prices_train);


# From the above boxplot, it is noticed there is almost zero correlation between Year of sales and sales Price, so it shouldn't be considered in our model

# In[ ]:


# MiscVal has the second lowest correlation with SalePrice
sns.lmplot (x="MiscVal",y="SalePrice",data=house_prices_train);


# From the above scatter plot, it is noticed there is almost zero correlation between Value of miscellaneous feature and sales Price, so it shouldn't be considered in our model

# In[ ]:


# observe the distributation for each column
#get the z score for each column and plot it
numerical_cols = house_prices_train.dtypes[house_prices_train.dtypes !="object"].index
for i in numerical_cols:
    ax = house_prices_train[i].plot(x='ZScore', y='FreqDist', kind='kde', figsize=(10, 6),title=i)
    plt.show()
    print ("\nZ-score for",i,":","\n", stats.zscore(house_prices_train[i]))
    print("mean=",house_prices_train[i].mean())
    print("skew=",house_prices_train[i].skew())


# From the plot we notice most columns have a positive skew and z score above the mean
# 
# 

# ## 2.3 Correlation Analysis

# In[ ]:


# Correlation of all numeric features in the training dataset
cor_train=house_prices_train.corr()
plt.figure(figsize=(29,19))
sns.heatmap(cor_train,annot=True,cmap="BrBG",square=True, annot_kws={'size': 8})


# In[ ]:


#box plot to determine the outliers in all the features
#get only numerical columns
stand_df_houses=house_prices_train._get_numeric_data()

#scale all columns
stand_df_houses=StandardScaler().fit_transform(stand_df_houses)

#plot to see the outliers
fig = plt.figure(figsize=(10,10))
ax = fig.gca()

sns.boxplot(data=stand_df_houses, orient='h', fliersize=2, linewidth=3, notch=True,
                 saturation=0.5, ax=ax)

ax.set_title(' Outliers in All features \n')
plt.show()


# ##### From the plot above we can see there are a lot of outliers, but we can't drop all of them because it will affect the results of the models

# In[ ]:


# Plotting Correlation of all numeric features in the training dataset

# Plot Outline
rows = 10
cols = 4
sorted_cols = cor_train.nlargest(len(cor_train), 'SalePrice')['SalePrice'].index #to sort columns from highest correlation with SalePrice
#Number_numerical = house_prices_train.dtypes[house_prices_train.dtypes !="object"].index
numerical_cols= len(sorted_cols)
fig, axs = plt.subplots(rows, cols, figsize=(cols*3,rows*3))
# Loop Definition
for r in range(0,rows):
    for c in range(0,cols):
        # Numerical Columns Condition
        i = r*cols+c
        if i < numerical_cols:
            sns.regplot(house_prices_train[sorted_cols[i]], house_prices_train["SalePrice"], ax = axs[r][c])
            # Correlation Definition
            correlation = house_prices_train[sorted_cols[i]].corr(house_prices_train["SalePrice"])
            corr_abs = round(abs(correlation),2)
            
            
            
            # Adjusting the plot appearance
            title = "r = " + "{}".format(corr_abs)
            axs[r][c].set_title(title,fontsize=10)
plt.tight_layout()
plt.show()


# In[ ]:


# Top Numerical Correlations with SalePrice

# get the only columns that have higher correlation than the threshold 
#select strong positive using threshold more than.4

threshold=0.4 
cor_train=house_prices_train.corr()
high_corre = cor_train.index[abs(cor_train["SalePrice"])>threshold]

#to sort columns from highest correlation with SalePrice
sorted_cols = cor_train.nlargest(len(high_corre),
'SalePrice')['SalePrice'].index 

plt.figure(figsize=(15,13))
sns.set(font_scale=1.5)

#plot heatmap with only the top features
nr_corr_matrix = sns.heatmap(house_prices_train[sorted_cols].corr(),
annot=True,cmap="BrBG",square=True, annot_kws={'size':14})


# In[ ]:


# Overallquality column has the highest correlation with the SalePrice
plt.subplot(1,1,1)

sns.barplot(house_prices_train.OverallQual,house_prices_train.SalePrice)
sns.lmplot (x="OverallQual",y="SalePrice",data=house_prices_train);


# Each OverallQual repreasent Rate of the material and finish of the house. From the plot above, If the house gets a high rate the price will increase

# In[ ]:


# GrLivArea column has the second high correlation with SalePrice
sns.lmplot (x="GrLivArea",y="SalePrice",data=house_prices_train)
plt.show()


# The GrLiveArea is the size of the living area in square feet. the bigger the living area the higher sales prices
# 
# 

# In[ ]:


# GarageCars column has the third correlation with SalePrice
sns.lmplot (x="GarageCars",y="SalePrice",data=house_prices_train)
sns.barplot(house_prices_train.GarageCars,house_prices_train.SalePrice)
plt.show()


# Category with garage car size 4 got a less sales prices than the other categories. further investigation will be done in the next section

# # 3. Data Cleaning

# ### 3.1 Training Dataframe

# In[ ]:


# check the precentage of the nan values in all columns before cleaning
house_prices_train.isnull().sum()


# Some columns in both dataframes contain categorical values. For some of these categories there is a category called NA (it doesn't mean nulls value) so we will change it through the function below
# 
# We will define function to replace all the np.nan (which are another type of categoricals but enter it as Na) from specific columns, to more meaningful value or string
# 

# In[ ]:


#function to replace np.nan with string'None' 
#takes columns list name and the datafrme name
def replace_none(column_name_list,dataframe):
    
    for i in column_name_list:
        dataframe[i].replace(np.nan,'None',inplace=True)
    return dataframe



#name of column will we change
list_column=['Alley','MiscFeature','PoolQC','BsmtQual','BsmtCond','BsmtExposure',
             'BsmtFinType1','BsmtFinType2','GarageType','GarageFinish',
              'GarageQual','GarageCond','PoolQC','Fence','FireplaceQu']
    
replace_none(list_column,house_prices_train)

#check on one of the column
house_prices_train.BsmtFinType1.unique()


# In[ ]:


house_prices_train.shape


# ##### line of code to help get the column index for both train and test dataframe
# column_index=np.argmax(np.array(DataframeName.columns=='columnName'))
# 

# In[ ]:


#numerical columns index to replace each missing data with mode/most_frequent

column_index_train=[3,25,26,6,74,72,30,31,32,33,35,42,58,60,63,64,72,73,57]


# In[ ]:


from sklearn.impute import SimpleImputer
#replace np.nan in all columns with most frequent value
imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imr = imr.fit(house_prices_train.iloc[:,column_index_train])

house_prices_train.iloc[:,column_index_train] = imr.transform(house_prices_train.iloc[:,column_index_train])


# In[ ]:


#see the relation between both years
house_prices_train[['GarageYrBlt','YearBuilt']].head()


#  From observation we can see that most of the garages are built in the same years of the houses
# 
# so we will replace the nan values in the garage year built to the value of the house year built in the same row

# In[ ]:


#replace each nan values in garage year with the year of the building in the same row

house_prices_train['GarageYrBlt'] = house_prices_train.apply(
lambda row: row['YearBuilt'] if np.isnan(row['GarageYrBlt']) else row['GarageYrBlt'],axis=1)


# In[ ]:


# check the precentage of the nan values in all columns after cleaning

houses_missing_value = house_prices_train.isnull().sum().sort_values(ascending=False)

houses_missing_percent = (house_prices_train.isnull().sum()/house_prices_train.isnull().count()).sort_values(ascending=False)

missing = pd.concat([houses_missing_value, houses_missing_percent], axis=1, keys=['Value', 'Percent'])

missing.head(6)


# In[ ]:


#drop ID column,because we can't use it on model
house_prices_train.drop("Id", axis = 1, inplace = True)


# ### 3.2 Cleaning Outliers From Top Features

# Since we can't delete all the outliers, we will check the top features to see the unusual data behavior, and drop the rows of these data 

# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x=house_prices_train['OverallQual'], y=house_prices_train['SalePrice'])


# 
# OverallQual outliers looks normal

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x =house_prices_train['GrLivArea'], y = house_prices_train['SalePrice'],c='darkred')
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see here that some data in GrLivArea are bigger than the rest, but they have lower SalePrice. That is unsual behavior and we need to drop them

# In[ ]:


#Deleting outliers from GrLivArea
house_prices_train = house_prices_train.drop(house_prices_train[(house_prices_train['GrLivArea']>4000) & (house_prices_train['SalePrice']<300000)].index).reset_index(drop=True)


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x=house_prices_train['GarageCars'], y=house_prices_train['SalePrice']);


# After observation the data in GarageCars, we notice unsual behavior on category 4.
# 
# Note: deleting this category has caused the RMSE to rise than before,so we decided to keep it
# 

# _________________________________

# To get the most possible minimum RMSE score we leave the rest of the outliers as they are,because from practicing we found dropping any more outliers can cause the RMSE to rise

# ### 3.3 Testing Dataframe

# In[ ]:


#call the function replace_none to replace Nan categorical to more appropiate value
replace_none(list_column,house_prices_test)


#  From observation we can see that most of the garages are built in the same years of the houses
# 
# so we will replace the nan values in the garage year built to the value of the house year built in the same row

# In[ ]:


#replace each nan values in garage year with the year of the building in the same row

house_prices_test['GarageYrBlt'] = house_prices_test.apply(
lambda row: row['YearBuilt'] if np.isnan(row['GarageYrBlt']) else row['GarageYrBlt'],axis=1)


# In[ ]:


# change 'None' category string to more meaningful string
house_prices_test['MasVnrType'].replace('None','no masonry vnr',inplace=True)


# ##### Line of code to help get the column index for both train and test dataframe
# column_index=np.argmax(np.array(DataframeName.columns=='columnName'))
# 

# In[ ]:


#numerical columns index to replace each missing data with mode/most_frequent


column_index=[2,3,9,23,24,25,26,34,36,37,38,47,48,53,55,60,61,62,77,78]

from sklearn.impute import SimpleImputer

imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imr = imr.fit(house_prices_test.iloc[:,column_index])

house_prices_test.iloc[:,column_index] = imr.transform(house_prices_test.iloc[:,column_index])


# In[ ]:


# check the precentage of the nan values in all columns after cleaning


houses_missing_value_test = house_prices_test.isnull().sum().sort_values(ascending=False)

houses_missing_percent_test = (house_prices_test.isnull().sum()/house_prices_test.isnull().count()).sort_values(ascending=False)

missing_T = pd.concat([houses_missing_value_test, houses_missing_percent_test], axis=1, keys=['Value', 'Percent'])

missing_T.head(10)


# In[ ]:


#drop ID column, because we can't use it on models
house_prices_test.drop("Id", axis = 1, inplace = True)


# There are some columns are regonized as numerical columns, but after  observing them, we noticed these columns should be categorical rather than numerical, so we will change them
# 
# 

# In[ ]:



house_prices_test['MSSubClass'] = house_prices_train['MSSubClass'].apply(str)

house_prices_test['YrSold'] = house_prices_train['YrSold'].astype(str)

house_prices_test['MoSold'] = house_prices_train['MoSold'].astype(str)


# There are some columns are regonized as numerical columns, but after  observing them, we noticed these columns should be categorical rather than numerical, so we will change them
# 
# 
# 

# In[ ]:



house_prices_test['MSSubClass'] = house_prices_test['MSSubClass'].apply(str)

house_prices_test['YrSold'] = house_prices_test['YrSold'].astype(str)

house_prices_test['MoSold'] = house_prices_test['MoSold'].astype(str)


# # 4. Dummies

# convert categorical variables to "one-hot encoded" using .get_dummies

# In[ ]:


house_prices_train=pd.get_dummies(house_prices_train, columns=['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition','YrSold','MoSold'], drop_first=True)


# In[ ]:


house_prices_test=pd.get_dummies(house_prices_test, columns=['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition','YrSold','MoSold'], drop_first=True)



# ## 4.1 find categorial features

# Compare the categoricals between the train and test dataframes. If we find a categorical column in one dataframe but not in the other, we will add the missing column with value of zero

# In[ ]:



# Get missing columns in the training test
missing_cols = set( house_prices_train.columns ) - set( house_prices_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    house_prices_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
house_prices_test = house_prices_test[house_prices_train.columns]


# In[ ]:


#drop the salePrice from test dataframe
house_prices_test.drop(['SalePrice'],inplace=True,axis=1)
test_X_=house_prices_test


# In[ ]:


# select the features as X
X=house_prices_train.drop(["SalePrice"],axis=1).copy()

#select the target as y
y=house_prices_train["SalePrice"].copy()


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.20, random_state=2) 


# # 5. Feature selection

# In[ ]:


#get the correlation for training 
cor_train_corr=house_prices_train.corr()


# In[ ]:



#we put threshold to determine which features should be selected 

#select strong positive using threshold more than 0.4
threshold_P=0.4
high_positive_corre=cor_train_corr['SalePrice']

#compare correlation with threshold
positives=(high_positive_corre[high_positive_corre>threshold_P])
positive_corre=pd.DataFrame(positives)
positive_corre.columns=positive_corre.columns.rename("positives")


#select strong negative using threshold less than 0.4
threshold_N=-0.4
high_negative_corre=cor_train_corr['SalePrice']

#compare correlation with threshold
negatives=high_negative_corre[high_negative_corre<threshold_N]
negative_corre=pd.DataFrame(negatives)
negative_corre.columns=negative_corre.columns.rename("negative")


# In[ ]:


positive_corre.sort_values(by='SalePrice',ascending=False)


# In[ ]:


negative_corre.sort_values(by='SalePrice',ascending=False)


# In[ ]:


feature_select=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath','YearBuilt','YearRemodAdd','MasVnrArea','Fireplaces','Foundation_PConc',
'ExterQual_Gd','BsmtFinType1_GLQ','Neighborhood_NridgHt','SaleType_New','SaleCondition_Partial','FireplaceQu_Gd',
'GarageType_Attchd','MasVnrType_Stone','Neighborhood_NoRidge','KitchenQual_Gd','Exterior2nd_VinylSd',
'Exterior1st_VinylSd','BsmtExposure_Gd','ExterQual_TA','KitchenQual_TA','BsmtQual_TA','GarageFinish_Unf']


# In[ ]:


X_FS=house_prices_train[feature_select]
    
y_FS=house_prices_train["SalePrice"]

X_train_FS, X_test_FS, y_train_FS, y_test_FS = train_test_split(
  X_FS, y_FS, test_size=0.20, random_state=2) 


test_X_FS=house_prices_test[feature_select]


# Note: you can try all the models, but for fast commit I make it most of them in comments

# # 6. Machine Learning

# For the model, we will try both models with standard scalar and without it to see the differences in performance

# ## 6.1 Lasso

# ### 6.1.1 With Scalar Features

# In[ ]:


# standard_scalar = StandardScaler()

# #standarlize X train
# X_train_ss = standard_scalar.fit_transform(X_train)
# X_train_ss = pd.DataFrame(X_train_ss, columns=X.columns)

# # #standarlize X test
# X_test_ss = standard_scalar.transform(X_test)
# X_test_ss = pd.DataFrame(X_test_ss, columns=X.columns)

# #standarlize the X to get the prediction  it
# test_X_ss = standard_scalar.transform(test_X_)
# test_X_ss = pd.DataFrame(X_test_ss, columns=X.columns)


# In[ ]:


# range_lasso=np.arange(0.01,100,0.01)


# model_lasso = LassoCV(alphas =range_lasso)

# #train the lasso model
# model_lasso.fit(X_train_ss,y_train)

# # get the train score
# model_lasso.score(X_train_ss,y_train)


# In[ ]:


# model_lasso.score(X_test_ss,y_test)


# In[ ]:


# # Perform 2-fold cross validation on lasso model
# scores = cross_val_score(model_lasso,X_train_ss,y_train, cv=5)
# print("Cross-validated training scores for lassoCV model:", scores.mean())


# In[ ]:


# #get the prediction of X from the test dataframe 
# pred_lasso=model_lasso.predict(test_X_)
# pred_lasso


# In[ ]:





# In[ ]:


# #save the prediction to submit and get the score of RMSE

# y_dataframe=pd.DataFrame(pred_lasso,columns=['SalePrice_pred_lasso'])

# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_lasso.csv',index=False)


# The Score of RMSE for LassoCV model is: 6.90683

# ## 6.2 Ridge

# ### 6.2.1 Without Scalar Features

# In[ ]:


# from sklearn.linear_model import RidgeCV,LassoCV

# range_Ridge=np.arange(0.01,100,0.01)
# ridgecv = RidgeCV(alphas=range_Ridge)
# # train ridgecv
# ridgecv.fit(X_train, y_train)

# #get the best alpha
# print('The best Alpha:',ridgecv.alpha_)


# In[ ]:


# #get the train score
# ridgecv.score(X_train, y_train)


# In[ ]:


# #get the test score
# ridgecv.score(X_test, y_test)


# In[ ]:


# #get the predict of the test dataframe
# pred_Ridge=ridgecv.predict(test_X_)
# pred_Ridge


# In[ ]:


# Basline_ridge=1 - pred_Ridge.mean()
# print('Baseline score for the test data dataframe:',Basline_ridge)


# In[ ]:



# # Perform 2-fold cross validation on RidgeCV model
# scores = cross_val_score(ridgecv,X_train,y_train, cv=5)
# print("Cross-validated training scores for RidgeCV model:", scores.mean())


# In[ ]:


# #save the prediction to submit and get the score of RMSE

# y_dataframe=pd.DataFrame(pred_Ridge,columns=['SalePrice_pred_ridge'])

# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_ridge.csv',index=False)


# <b><I>The score of RMSE for RidgeCV model: 0.42697</b></I>

# ### 6.2.2 With Scalar Features

# In[ ]:


# range_Ridge=np.arange(0.01,100,0.01)
# ridgecv_s = RidgeCV(alphas=range_Ridge)

# ridgecv_s.fit(X_train_ss, y_train)

# #get the best alpha
# print('The best Alpha:',ridgecv_s.alpha_)


# In[ ]:


# ridgecv_s.score(X_train_ss, y_train)


# In[ ]:


# ridgecv_s.score(X_test_ss,y_test)


# In[ ]:


# pred_Ridge_s=ridgecv_s.predict(test_X_)
# pred_Ridge_s


# In[ ]:



# # Perform 2-fold cross validation on lasso model
# scores = cross_val_score(ridgecv_s,X_train_ss,y_train, cv=5)
# print("Cross-validated training scores for RidgeCV model:", scores.mean())


# In[ ]:


# #save the prediction to submit and get the score of RMSE

# y_dataframe=pd.DataFrame(pred_Ridge_s,columns=['SalePrice_pred_ridge_s'])

# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_ridge_s.csv',index=False)


# <b><I>The score of RMSE for RidgeCV model with standarize is: 6.64239</b></I>

# ## 6.3 Bagging Regressor
# 

# ### 6.3.1 Without Scalar Features

# In[ ]:


# param_reg = { 'max_features': [0.3,.4,.5, 0.6,.7,.8, 1],
#         'n_estimators': [50,100, 150, 200], 
#          'base_estimator__max_depth': [3, 5,10,15, 20]}


# In[ ]:


# from sklearn.tree import DecisionTreeRegressor

# model_reg=BaggingRegressor(base_estimator=DecisionTreeRegressor(),)

# model_GSsearch = GridSearchCV(model_reg,param_reg, cv=9)
# model_GSsearch.fit(X_train, y_train)


# In[ ]:


# model_GSsearch.best_params_


# In[ ]:


# model_GSsearch.score(X_train, y_train)


# In[ ]:


# model_GSsearch.score(X_test, y_test)


# In[ ]:


# pred_bagging=model_GSsearch.predict(test_X_)
# pred_bagging


# In[ ]:


# scores_bagg = cross_val_score(model_GSsearch,X_train, y_train, cv=6)
# print("Mean of Cross-validated scores:",scores_bagg.mean())


# In[ ]:


#save the prediction to submit and get the score of RMSE

# y_dataframe=pd.DataFrame(pred_bagging,columns=['SalePrice_pred_bagging'])

# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_bagging.csv',index=False)


# <b><I>The score of the Bagging Model is: 0.14994</b></I>

# ### 6.3.2 With Scalar Features

# In[ ]:



# model_reg_s=BaggingRegressor(base_estimator=DecisionTreeRegressor(),)

# model_GSsearch_s = GridSearchCV(model_reg_s,param_reg, cv=9)
# model_GSsearch_s.fit(X_train_ss, y_train)


# In[ ]:


model_GSsearch_s.best_params_


# In[ ]:


# model_GSsearch_s.score(X_train_ss, y_train)


# In[ ]:


# model_GSsearch_s.score(X_test_ss, y_test)


# In[ ]:


# pred_bagging_s=model_GSsearch_s.predict(test_X_ss)
# pred_bagging_s


# In[ ]:


# scores_bagg = cross_val_score(model_GSsearch_s,X_train_ss, y_train, cv=6)
# print("Mean of Cross-validated scores:",scores_bagg.mean())


# In[ ]:


#save the prediction to submit and get the score of RMSE

# y_dataframe=pd.DataFrame(pred_bagging_s,columns=['SalePrice_pred_bagging_s'])
# y_dataframe = y_dataframe[:-1]
# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_bagging_s.csv',index=False)


# <b><I>The score of Bagging Regressor with standarize is: 0.558</b></I>

# ## 6.4 Random forest

# ### 6.4.1 Without Scalar Features

# In[ ]:


param_Rforest = { 'max_depth': [ 2, 3, 4, 5, 6,9,13,15,20],
          'max_features':[.1,.2,.3,.4,.6,.7,.8,.9,1],
          'n_estimators': [20,50,130,150],
          'min_samples_leaf': [1, 2, 3, 4]
          }


# In[ ]:


model_Rforest = RandomForestRegressor(random_state=1)
Rforest_GSsearch = GridSearchCV(model_Rforest ,param_Rforest, cv=6, verbose=1, n_jobs=-1 )

Rforest_GSsearch.fit(X_train, y_train)

train_score_RF=Rforest_GSsearch.score(X_train, y_train)

test_score_RF=Rforest_GSsearch.score(X_test, y_test)
 

print("train score for Random Forest model:",train_score_RF)
print("test score for Random Forest model:",test_score_RF)


# In[ ]:


y_pred_forest=Rforest_GSsearch.predict(test_X_)
y_pred_forest


# In[ ]:



y_dataframe=pd.DataFrame(y_pred_forest,columns=['SalePrice_pred'])

dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
dataset.columns=['Id','SalePrice']
dataset.to_csv('sample_submission_forest.csv',index=False)


# <b><I>The RMSE score for Random forest model is: 0.148</b></I>

# ### 6.4.2 With Scalar Features

# In[ ]:


# model_Rforest_s = RandomForestRegressor(random_state=1)

# Rforest_GSsearch_s = GridSearchCV(model_Rforest_s ,param_Rforest, cv=6, n_jobs=-1 )

# Rforest_GSsearch_s.fit(X_train_ss, y_train)

# train_score_s=Rforest_GSsearch_s.score(X_train_ss, y_train)

# test_score_s=Rforest_GSsearch_s.score(X_test_ss, y_test)
 

# print("train score for Random Forest Scaled :",train_score_s)
# print("test score  score for Random Forest Scaled :",test_score_s)


# In[ ]:


# y_pred_forest_s=Rforest_GSsearch_s.predict(test_X_)
# y_pred_forest_s


# In[ ]:



# y_dataframe=pd.DataFrame(y_pred_forest_s,columns=['SalePrice_pred_forest'])
# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_forest_scale.csv',index=False)


# <b><I>The RMSE score for Random forest model with standarize is: 1.15475</b></I>

# ## 6.5 Decision Tree Regressor

# ### 6.5.1 Without Scalar Features

# In[ ]:


# param_DTress = { 'max_depth': [1, 2, 3, 4, 5, 6,9,13,15],
#           'max_features':[.1,.2,.3,.4,.6,.7,.8],
#           'max_leaf_nodes': [5, 6, 7, 8, 9, 10],
#           'min_samples_leaf': [1, 2, 3, 4]
#           }


# In[ ]:



# model_DTree = DecisionTreeRegressor( random_state=1)

# DTress_GSsearch = GridSearchCV(model_DTree,param_DTress, cv=6, n_jobs=-1 )

# DTress_GSsearch.fit(X_train, y_train)

# train_scores=DTress_GSsearch.score(X_train, y_train)


# test_scores=DTress_GSsearch.score(X_test, y_test)

# print('Train score for Decision Tree without standarize: ',train_scores)

# print('Test score for Decision Tree without standarize: ',test_scores)


# In[ ]:


# scores_DTress = cross_val_score(DTress_GSsearch,X_test, y_test, cv=5)
# print("cross validation:",scores_DTress.mean())


# In[ ]:


# y_pred_tres=DTress_GSsearch.predict(test_X_)
# y_pred_tres


# In[ ]:


# y_dataframe=pd.DataFrame(y_pred_tres,columns=['SalePrice_pred'])

# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_DT.csv',index=False)


# <b><I>The Score of RMSE for Decision Tree Regressor model is: 0.25538</b></I>

# ### 6.5.2 With Scalar Features

# In[ ]:



# model_DTree_s = DecisionTreeRegressor( random_state=1)

# DTress_GSsearch_s = GridSearchCV(model_DTree_s,param_DTress, cv=6, n_jobs=-1 )

# DTress_GSsearch_s.fit(X_train_ss, y_train)

# #get the train score
# train_scores_DT=DTress_GSsearch_s.score(X_train_ss, y_train)

# #get the test score
# test_scores_DT=DTress_GSsearch_s.score(X_test_ss, y_test)

# print('Train score for Decision Tree with standarize: ',train_scores_DT)

# print('Test score for Decision Tree with standarize: ',test_scores_DT)


# In[ ]:


# #cross validation for training the data
# scores_DTress = cross_val_score(DTress_GSsearch,X_train_ss, y_train, cv=5)
# print("cross validation for training Decision Tree:",scores_DTress.mean())


# In[ ]:


# #predict the test X 
# y_pred_tres_s=DTress_GSsearch_s.predict(test_X_)
# y_pred_tres_s


# In[ ]:


# #add the prediction to submission file

# y_dataframe=pd.DataFrame(y_pred_tres_s,columns=['SalePrice_pred'])
# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_DT_S.csv',index=False)


# <b><I>The score of Decision Tree model with standarize is: 1.07622</b></I>

# ### calculate the average of more than one model prediction

# In[ ]:


# models_avg = (pred_Ridge+pred_bagging) / 2
# models_avg


# In[ ]:


# y_dataframe=pd.DataFrame(models_avg,columns=['average_pred'])
# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_average.csv',index=False)


# <b><I>The average score of the two models: Ridge and Bagging:0.13164</b></I>

# _______________________________

# In[ ]:


# models_avg_2 = (pred_Ridge+y_pred_tres) / 2

# y_dataframe=pd.DataFrame(models_avg_2,columns=['average_pred'])
# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_average_2.csv',index=False)



# <b><I>The average score of the two models: Ridge and Decision Tree:0.14914</b></I>

# ## 6.6 Models with Feature Selection

# ### Random Forest

# In[ ]:


param_Rforest_fs = { 'max_depth': [ 2, 3, 4, 5, 6,9,13,15,20],
          'max_features':[.1,.2,.3,.4,.6,.7,.8,.9,1],
          'n_estimators': [20,50,130,150],
          'min_samples_leaf': [1, 2, 3, 4]
          }


# In[ ]:


# model_Rforest_selection = RandomForestRegressor(random_state=2)

# Rforest_GSsearch_selection = GridSearchCV(model_Rforest_selection ,
# param_Rforest_fs,cv=6, n_jobs=-1 )

# Rforest_GSsearch_selection.fit(X_train_FS, y_train_FS)

# train_score_RF_selection=Rforest_GSsearch_selection.score(X_train_FS, y_train_FS)

# test_score_RF_selection=Rforest_GSsearch_selection.score(X_test_FS, y_test_FS)
 
# print("train score for Random Forest model with selection features:",
#       train_score_RF_selection)
# print("test score for Random Forest model  with selection features:",
#       test_score_RF_selection)


# In[ ]:


# y_pred_forest_FS=Rforest_GSsearch_selection.predict(test_X_FS)

# y_dataframe=pd.DataFrame(y_pred_forest_FS,columns=['RandomF_pred'])
# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_RForesr_FS.csv',index=False)


# <b><I>The RMSE score with Feature Selection: 0.15814</b></I>

# ### Baggin regressor with Feature Selection

# In[ ]:


param_reg_fs = { 'max_features': [0.3,.4,.5, 0.6,.7,.8, 1],
        'n_estimators': [50,100, 150, 200], 
         'base_estimator__max_depth': [3, 5,10,15, 20]}


# In[ ]:


# model_reg_fs=BaggingRegressor(base_estimator=DecisionTreeRegressor(),)

# model_GSsearch_fs = GridSearchCV(model_reg_fs,param_reg_fs, cv=9)
# model_GSsearch_fs.fit(X_train_FS, y_train_FS)

# train_score_reg_selection=model_GSsearch_fs.score(X_train_FS, y_train_FS)

# # test_score_reg_selection=model_GSsearch_fs.score(X_test_FS, y_test_FS)
 
# print("train score for Bagging model with selection features:",
#       train_score_reg_selection)
# print("test score for Bagging model  with selection features:",
#       test_score_reg_selection)


# In[ ]:


# pred_bagging_fs=model_GSsearch_fs.predict(test_X_FS)



# y_dataframe=pd.DataFrame(pred_bagging_fs,columns=['RandomF_pred'])
# dataset=pd.concat([sampl_sub['Id'],y_dataframe],axis=1)
# dataset.columns=['Id','SalePrice']
# dataset.to_csv('sample_submission_bagg_FS.csv',index=False)


# <b><I>The RMSE score with Feature Selection: 0.15689</b></I>

# # The Final Result

# After testing five models:<b> LassCV,RidgeCV,Bagging Regressor,Random Forest Regressor, and Decision Tree</b>, with both Standarization and feature selection.Using score,RMSE and cross validation metrics.<br> 
# 
# The first best RMSE score is form the average of two models<b> Ridge and Bagging regressor</b> with a score<b> 0.131</b>. The secound best model give us RMSE score with<b> 0.148 </b>is the Random Forest Tree, without using standarization or a feature selection. 
# 
# Both models have more ability to react and make a good prediction to new data it will get. In a simple word these models can generalize a new data

# # Conclusion and recommendations
# 
# By studying the effects of features on the houses sales prices, We investigated, cleaned and visualized 80 features in both training and testing datasets. 
# After that we used the dummies to convert categorical variables into indicator 0 or 1. 
# Then we used Correlation for feature selection to find the strong positive and also the strong negative features ,and tried to use them with models in order to give us the best RMSE score. 
# The Best RMSE score we got was 0.131 came from the average of two models ;Ridge and Bagging regressor models, by training all the features without scaling.
# This research will help both the sellers and the buyers to ensure they know the correct house price through the prediction in more intelligent way.
# Based on the conclusion we recommend:
# - Modeling the data on more advanced models to get the possible minimum RMSE score
# - Nowadays, a lot of houses are designed with more intelligent security, so it will be interesting to collect this kind of new feature to the other features we already have

# In[ ]:




