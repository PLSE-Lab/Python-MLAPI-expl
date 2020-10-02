#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques

# ## Table of Contents

# 1. [Introduction](#intro)<br><br>
# 2. [Load and Check Data](#load_and_check_data)<br><br>
# 3. [Data Exploration and Preprocessing](#data_exp_pro)<br>- [3.1 Outliers](#outliers) <br>- [3.1.1 Outliers Detection](#outliers_detection)<br>- [3.1.2 Removal of Outliers](#r_outliers)<br>- [3.2 Target Variable](#target_variable)<br>- [3.2.1 Distribution](#t_distribution)<br>- [3.2.2 Transformation](#t_transformation)<br>- [3.3 Correlation Check](#correlation)<br>- [3.4 Pairplot](#pairplot)<br>- [3.5 Seperation](#seperation)<br>- [3.6 Concatenation of train and test datasets](#concat)<br><br>
# 4. [Feature Engineering](#feature_engineering)<br>- [4.1 Missing Values](#missing_values)<br>- [4.1.1 Check For Missing Values](#check_missing)<br>- [4.1.2 Imputing Missing Values](#imputing_missing)<br>- [4.2 Transformation & Encoding](#transformation_encoding)<br>- [4.2.1 Data Type & LabelEncoder](#data_type)<br>- [4.2.2 New Feature](#new_feature)<br>- [4.2.3 Skewed Numeric Features](#numeric_features)<br>- [4.2.4 Log1p Transformation](#log1p)<br>- [4.2.5 Dummy Variables](#dummy)<br>- [4.2.6 Train Test Split](split)
# <br><br>
# 5. [Modeling](#modeling)<br>- [5.1 Feature Scaling](#feature_scaling)<br>- [5.2 Simple Modeling](#simple_modeling)<br>- [5.3 Stacked Regression and GridSearch](#stacked_regression_and_gridsearch)<br> - [5.4 Ensemble - Averaging](#avg)
# <br><br>
# 6. [Submission](#submission)<br><br>

# ## 1. Introduction
# <a id='intro'></a>

# The goal of this project is to predict the house sales prices in Ames, Iowa. The first part consists of various visualization and normalization. By visualizing the distribution of features, we can gain some insight how to handle the data and determine the direction of the project. Also the distribution of target variable is transformed to normal by taking logarithm function so that they can fit many linear models better and lead better performance. 
# 
# The second part is feature engineering. It comprises filling missing values, encoding and transformation of skewed features. Reading the original document written by the dataset creater is very important for the step for filling missing values since there are some useful directions how to deal with missing values. Specifically, it will help you when to fill the missing values with zero instead of None or mode (the most frequent value). The next thing to do is encoding. Dataset must be transformed to appropriate format that a machine learning model can handle. By using LabelEncoder and get_dummies, we can convert them into a numerical representation that we can apply our machine learning algorithms to. Once we finish the encoding process, some highly skewed features need to be transformed to normal as the target variable was transformed.
# 
# Lastly, in modeling part, there are two things to notice: stacking and ensemble (averaging). Stacking boosts accuracy of our results predicted by many single models; however, we can boost the accuracy even more by averaging the prediction of stacking and predictions of best models. At the end, I managed to score 0.11948 which is ranked top 17 percent in the competition.

# In[ ]:





# ## 2. Load and Check Data
# <a id='load_and_check_data'></a>

# In[ ]:


# import libraries 

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import scipy.stats as stats
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# let's start by reading the train and test dataset.

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


# print out the first five rows of the train datasets

train_df.head()


# In[ ]:


# print out the first five rows of the test datasets.

test_df.head()


# In[ ]:





# ## 3. Data Exploration & Preprocessing
# <a id='data_exp_pro'></a>

# ### 3.1 Outliers
# <a id='outliers'></a>
# 
# As [the data document](http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt) mentioned, it is recommended removing any houses with more than 4000 square feet from the data set.

# #### 3.1.1 Outliers Detection
# <a id='outliers_detection'></a>

# In[ ]:


plt.figure()
otl = sns.lmplot('GrLivArea', 'SalePrice',data=train_df, fit_reg=False);


# You can clearly see two outliers at the bottom right. Compared to the size of the house, they are extremely cheap.

# In[ ]:


#train_df[(train_df['SalePrice'] < 300000) & (train_df['GrLivArea'] > 4000)]
train_df[(train_df['GrLivArea'] > 4000)][['SalePrice','GrLivArea']]


# #### 3.1.2 Removal of the Outliers
# <a id='r_outliers'></a>

# In[ ]:


#train_df.drop(train_df[(train_df['SalePrice'] < 300000) & (train_df['GrLivArea'] > 4000)].index,inplace=True)
train_df.drop(train_df[(train_df['GrLivArea'] > 4000)].index,inplace=True)

plt.figure()
sns.lmplot('GrLivArea', 'SalePrice',data=train_df, fit_reg=False);
plt.xlim(0,5500);
plt.ylim(0,800000);


# There could be many ways to handle outliers; however, I decide to follow the data docs author's recommendation (removing any houses with more than 4000 square feet from the data set). 

# In[ ]:


# check the dimensions
print(train_df.shape)


# In[ ]:





# ### 3.2 Target Variable
# <a id='target_variable'></a>

# #### 3.2.1 Distribution
# <a id='t_distribution'></a>

# In[ ]:


sns.distplot(train_df['SalePrice'])
plt.title('SalePrice Distribution')
plt.ylabel('Frequency')

plt.figure()
qq = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

# For normally distributed data, the skewness should be about zero. 
# A skenewss  value greater than zero means that there is more weight in the left tail of the distribution

print("Skewness: {:.3f}".format(train_df['SalePrice'].skew()))


# As you can see the distribution plot and qq plot, the target variable is skewed to the right. In order to use many general linear models, we need to transform it to normal.

# #### 3.2.2 Target Variable Transformation
# <a id='t_transformation'></a>

# In[ ]:


# log1p calculates log(1 + input)

train_df['SalePrice'] = np.log1p(train_df['SalePrice'])


# In[ ]:


# let's check the result of the transformation

sns.distplot(train_df['SalePrice'])
plt.title('SalePrice Distribution')
plt.ylabel('Frequency')

plt.figure()
qq = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

print("Skewness: {:.3f}".format(train_df['SalePrice'].skew()))


# By just taking log, the shape of the distribution becomes almost normal.  

# In[ ]:





# ### 3.3 Correlation Check
# <a id='correlation'></a>

# In[ ]:


plt.figure(figsize=(15,5))

# correlation table
corr_train = train_df.corr()

# select top 10 highly correlated variables with SalePrice
num = 10
col = corr_train.nlargest(num, 'SalePrice')['SalePrice'].index
coeff = np.corrcoef(train_df[col].values.T)

# heatmap
heatmp = sns.heatmap(coeff, annot = True, xticklabels = col.values, yticklabels = col.values, linewidth=2,cmap='PiYG', linecolor='blue')


# Based on the correlation table shown, we can conjecture that the features related with __quality__ (OverallQual,FullBath, YearBuilt, YearRemodAdd) and the __size__ (GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF) may play an important role in prediction.

# In[ ]:





# ### 3.4 Pairplot
# <a id='pairplot'></a>
# 
# One of the best way to visualize the relationship between the target variable and many features at the same time is pairplot. In our case, instead of plotting the whole features with target variable, only chose the top 10 most highly correlated features with target variable: 'SalesPrice','OverallQual',GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 1stFlrSF', 'FullBath','YearBuilt', 'YearRemodAdd'.

# In[ ]:


# Visualized the relationship between the target variable and top 10 features highly correlated with the target variable.

sns.pairplot(train_df[col], size=3);


# As we can see the first column of the plots, it is not perfectly linear but we can say some of them are showing some positive linear pattern.

# In[ ]:





# ### 3.5 Independent Variables (Id, SalePrice) Seperation
# <a id='seperation'></a>

# In[ ]:


# seperate id from datasets and drop them.

train_id = train_df.iloc[:,0]
test_id = test_df.iloc[:,0]

train_df.drop('Id',axis=1,inplace = True)
test_df.drop('Id',axis=1,inplace = True)


# In[ ]:


# seperate the target variable (SalePrice) from the train

y_df = train_df['SalePrice']
train_df.drop('SalePrice',axis=1,inplace=True)

print('dimension of the train:' , train_df.shape)
print('dimension of the test:' , test_df.shape)


# In[ ]:





# ### 3.6 Concatenation of train and test datasets
# <a id='concat'></a>

# In[ ]:


# In order to avoid repeating unnecessary codes, for our convenience, let's combine the train and test set.
df = pd.concat([train_df, test_df]).reset_index()

df.drop(['index'],axis=1,inplace=True)


# In[ ]:


print('dimension of the dataset:' , df.shape)
df.head()


# In[ ]:





# ## 4. Feature Engineering
# <a id='feature_engineering'></a>

# ### 4.1 Missing Values
# <a id='missing_values'></a>
# 
# Handling missing data is important as many machine learning algorithms do not support data with missing values.

# #### 4.1.1 Check For Missing Values
# <a id='check_missing'></a>
# 

# In[ ]:


mc = pd.DataFrame(df.isnull().sum(),columns=['Missing Count'])
mc = mc[mc['Missing Count']!=0]
mc['Missing %'] = (mc['Missing Count'] / df.shape[0]) * 100
mc.sort_values('Missing %',ascending=False)


# #### 4.1.2 Imputing Missing Values
# <a id='imputing_missing'></a>

# #### None

# In[ ]:


nones = ['PoolQC', 'MiscFeature', 'Alley','Fence', 'FireplaceQu', 'GarageType','GarageFinish',
        'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'MasVnrType']

for none in nones:
    df[none].fillna('None',inplace = True)
    


# The data documentation says missing values in the above features (the elements in the list "nones") mean these properties do not have one of them: garage, basment, fireplace, alley access, pool, misc features, fence, or masonry veneer. 

# In[ ]:





# #### Zero

# In[ ]:


zeros = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
         'BsmtFullBath','BsmtHalfBath','MasVnrArea']

for zero in zeros:
    df[zero].fillna(0, inplace = True)


# Some features are explicitly described that the missing values mean zero or we can assume that these properties have zero basement or garage or masonry veneer.

# In[ ]:





# #### Removal

# In[ ]:


Counter(df.Utilities)


# As you can see the above, there are only two categories in the Utilities feature: 'AllPub' and 'NoSeWa'. Moreover, except 1 'NoSeWa and 2 NaN values, all the values are 'AllPub' ,which means the data is very imbalanced and this feature seems not that helpful in predictive modeling. I decide to remove this feature.

# In[ ]:


df.drop('Utilities',axis=1, inplace=True)


# In[ ]:





# #### Mode
# 
# Unlike Utilities feature, the features below are not extremly imbalanced and consist of many categories. Also, there are only a few missing values in each feature. Therefore, we can fill missing values with most frequently occurred values.

# In[ ]:


freq = ['MSZoning','Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual','Functional']

for fr in freq:
    df[fr].fillna(df[fr].mode()[0], inplace=True)


# In[ ]:





# #### Groupby
# 
# Since there are a lot of missing values in the LotFrontage, simply filling in missing values with median or mode may affect badly our models. We can fill missing values in LotFrontage with the median LotFrontage of similar rows according to LotArea and Neighborhood.

# In[ ]:


df['old_lotfrontage'] = df['LotFrontage']

df['LotFrontage'] = df.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
ol = sns.distplot(df['old_lotfrontage'].dropna(),ax=ax1,kde=True,bins=70)
lf = sns.distplot(df['LotFrontage'],ax=ax2,kde=True,bins=70,color='red')

# drop the old_lotfrontage as we finished the comparison
df.drop('old_lotfrontage',axis=1,inplace=True)


# The left blue plot is the distribution of the 'LotFrontage' from original dataset after simply omitting missing values and the right red plot is the distribution of the 'LotFrontage' after filling missing values with the median ages of similar rows. As you can see, the distributions are very similar each other.

# In[ ]:


print("Remaining missing values:",df.isnull().sum().sum())


# In[ ]:





# ### 4.2 Transformation & Encoding
# <a id='transformation_encoding'></a>

# #### 4.2.1 Data Type & LabelEncoder
# <a id='data_type'></a>

# #### Nominal
# 
# Machine learning algorithms will require that nominal variables be converted into dummy variables (0 or 1) as all of scales are mutually exclusive (no overlap) and none of them have any numerical significance.

# In[ ]:


# get_dummies can convert data to 0 and 1 only if the data type is string. Among the many nominal features,
# MSSubClass, MoSold, and YrSold are integer type so we need to convert them to string type.

df['MoSold'] = df.astype(str)
df['YrSold'] = df.astype(str)
df['MSSubClass'] = df.astype(str)

nominals = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
           'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir','GarageType','MiscFeature','SaleType','SaleCondition','MoSold','YrSold']


# In[ ]:





# #### Ordinal
# 
# With ordinal scales, it is the order of the values is what's important and significant, but the differences between each one is not really known. Therefore, unlike nominal, ordinal values matter the order (e.g. a > b > c). In this case, instead of using get_dummies (0 or 1), it is better to use LabelEncoder.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

ordinals = ['LotShape','LandSlope','OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',
           'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','Electrical','KitchenQual',
            'Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence']

for ordinal in ordinals:
    le = LabelEncoder()
    le.fit(df[ordinal])
    df[ordinal] = le.transform(df[ordinal])


# In[ ]:





# #### 4.2.2 New Feature
# <a id='new_feature'></a>
# 
# As we have seen the correlation table, area features such as GrLivArea or BsmtSF are highly correlated with house sales price. Creating new feature regarding the total area may help predict the target variable.

# In[ ]:


# Total square feet of houses

df['totalArea'] = df['GrLivArea'] + df['TotalBsmtSF']


# In[ ]:





# #### 4.2.3 Skewed Numeric Features
# <a id='numeric_features'></a>

# In[ ]:


# Assign numeric features by excluding non numeric features
numeric = df.dtypes[df.dtypes != 'object'].index

# Display the skewness of each column and sort the values in descending order 
skewness = df[numeric].apply(lambda x: x.skew()).sort_values(ascending=False)

# Create a dataframe and show 5 most skewed features 
sk_df = pd.DataFrame(skewness,columns=['skewness'])
sk_df['skw'] = abs(sk_df)
sk_df.sort_values('skw',ascending=False).drop('skw',axis=1).head()


# - Skewnewss quantifies how symmetrical the distribution is.
# - If skewness is less than -1 or greater than 1, the distribution is highly skewed.
# - If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
# - If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# 
# Source: https://help.gooddata.com/display/doc/Normality+Testing+-+Skewness+and+Kurtosis

# In[ ]:





# #### 4.2.4 Log1p Transformation
# <a id='log1p'></a>

# In[ ]:


# As a general rule of thumb, skewness with an absolute value less than 0.5 is considered as a acceptable range of skewness for normal distribution of data
skw_feature = skewness[abs(skewness) > 0.5].index

# Transform skewed features to normal distribution by taking log(1 + input)
df[skw_feature] = np.log1p(df[skw_feature])


# Even though I consider the features with more than 0.5 skewness as not normally distributed feature, you can try some different numbers to improve the result since there is no clear cut rule for the cutoff value.  

# In[ ]:





# #### 4.2.5 Dummy Variable
# <a id='dummy'></a>
# 
# A dummy variable is one that takes the value 0 or 1 to indicate the absence or presence of some categorical effect that may be expected to shift the outcome. Since 'get_dummies' function does not affect numerical values, only nominal data type values will be converted to 0 or 1 (as we already converted ordinal types values to numeric by LabelEncoder, we do not have to worry about that). 

# In[ ]:


df = pd.get_dummies(df)
print(df.shape)


# In[ ]:





# #### 4.2.6 Train Test Split
# <a id='split'></a>
# 
# Since data cleaning and feature engineering process are finished, we need to split the combined dataset into train and test as given dataset

# In[ ]:


# Split the combined dataset into two: train and test

X_train = df[:train_df.shape[0]]
X_test = df[train_df.shape[0]:]

#X_train, X_test, y_train, y_test = train_test_split(df,y_df, random_state = 1)


# In[ ]:


print("training shape:{}, test shape:{}".format(X_train.shape,X_test.shape))


# In[ ]:





# ## 5. Modeling <a id='modeling'></a>

# In[ ]:


# Import libraries

from sklearn.model_selection import GridSearchCV,learning_curve, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.linear_model import LassoCV,ElasticNetCV,Lasso,ElasticNet
from sklearn.kernel_ridge import KernelRidge

from mlxtend.regressor import StackingRegressor
from xgboost import XGBRegressor


# In[ ]:


print(X_train.shape, X_test.shape,y_df.shape)


# In[ ]:





# ### 5.1 Feature Scaling 
# <a id='feature_scaling'></a>
# Many machine learning algorithms expect the scale of the input and even the output data to be equivalent. It can help in methods, particularly linear models, that weights in order to make a prediction. Among many scaling techniques, I decided to use StandardScaler since in the previous section 3.1.1 outlier detections, we have already removed the data whose GrLivArea is greater than 4000. Therefore, we would worry the information loss by removing the outliers more rather than distortion of the data shape because of outliers. If we did not delete those outliers, it would be better to choose RobustScaler.

# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
#X_train = RobustScaler().fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns = df.columns )

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns = df.columns)
#X_test = RobustScaler().fit_transform(X_test)


# In[ ]:


y_df.head()


# In[ ]:





# ### 5.2 Simple Modeling
# <a id='simple_modeling'></a>

# We will be evaluating below algorithms' root-mean-squared-error for train dataset. 
# 
# > - Ridge Regression
# > - Lasso Regression
# > - ElasticNet Regression
# > - Support Vector Machine
# > - Random Forest
# > - XG Boost

# #### Kfold

# In[ ]:


kfold = KFold(n_splits=20, random_state= 0, shuffle = True)


# In[ ]:





# #### Score

# In[ ]:


def rmsle_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_df, scoring="neg_mean_squared_error", cv = kfold))
    return(rmse)


# In[ ]:





# #### Kernel Ridge

# In[ ]:


KR = KernelRidge()

KR_param_grid = {
    'alpha' : [0.93],
    'kernel' : ['polynomial'],
    'gamma':[0.001],
    'degree': [3],
    'coef0': [1.5]
}

KR_CV = GridSearchCV(KR, param_grid = KR_param_grid, cv = kfold, scoring = "neg_mean_squared_error",n_jobs = -1, verbose = 1)
KR_CV.fit(X_train, y_df)
KR_best = KR_CV.best_estimator_
print(KR_best)

# scaler, cv = 20
# * KernelRidge(alpha=1.0, coef0=0.9, degree=2, gamma=0.004, kernel='polynomial',kernel_params=None)
# ** KernelRidge(alpha=0.93, coef0=1.5, degree=3, gamma=0.001, kernel='polynomial',kernel_params=None)
# *** KernelRidge(alpha=0.93, coef0=1.5, degree=3, gamma=0.001, kernel='polynomial',kernel_params=None) - 0.12514


# In[ ]:


y_submission_1 = np.expm1(KR_best.predict(X_test))


# In[ ]:


score = rmsle_cv(KR_best)
print("Kernel Ridge mean score:", score.mean())
print("Kernel Ridge std:", score.std())


# In[ ]:





# #### Lasso Regression

# In[ ]:


lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 0.4, 0.6, 0.8, 1, 1.2], random_state = 1, n_jobs = -1, verbose = 1)
lasso.fit(X_train, y_df)
alpha = lasso.alpha_
print("Optimized Alpha:", alpha)

lasso = LassoCV(alphas = alpha * np.linspace(0.5,1.5,20), cv = kfold, random_state = 1, n_jobs = -1)
lasso.fit(X_train, y_df)
alpha = lasso.alpha_
print("Final Alpha:", alpha)

# scaler cv = 20
#lasso = LassoCV(alphas = 0.00244736842105, cv = kfold, random_state = 1, n_jobs = -1, verbose = 1)
#lasso.fit(X_train, y_df)

#Final Alpha: 0.00244736842105


# In[ ]:


print("Lasso mean score:", rmsle_cv(lasso).mean())
print("Lasso std:", rmsle_cv(lasso).std())


# In[ ]:


y_submission_2 = np.expm1(lasso.predict(X_test))


# In[ ]:





# #### ElasticNet Regression

# In[ ]:


elnet = ElasticNetCV(alphas = [0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 0.4, 0.6, 0.8, 1, 1.2] 
                ,l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
                ,cv = kfold, random_state = 1, n_jobs = -1)
elnet.fit(X_train, y_df)
alpha = elnet.alpha_
ratio = elnet.l1_ratio_
print("Optimized Alpha:", alpha)
print("Optimized l1_ratio:", ratio)

elnet = ElasticNetCV(alphas = alpha * np.linspace(0.5,1.5,20), l1_ratio = ratio * np.linspace(0.9,1.3,6), 
                     cv = kfold, random_state = 1, n_jobs = -1)
elnet.fit(X_train, y_df)

alpha = elnet.alpha_
ratio = elnet.l1_ratio_

print("Final Alpha:", alpha)
print("Final l1_ratio:", ratio)

# scaler cv = 20
# Final Alpha: 0.0276315789474, Final l1_ratio: 0.09


# In[ ]:


print("ElasticNet mean score:", rmsle_cv(elnet).mean())
print("ElasticNet std:", rmsle_cv(elnet).std())


# In[ ]:


y_submission_3 = np.expm1(elnet.predict(X_test))
# kaggle_score: 0.12302


# In[ ]:





# #### Support Vector Machine

# In[ ]:



epsilons = [0.03]
degrees = [2]
coef0s = [1.6]

gammas = ['auto']
Cs = [0.1]
kernels = ['poly']

param_grid = dict(C=Cs, epsilon = epsilons, gamma=gammas, kernel=kernels, degree= degrees, coef0=coef0s)
SVMR = GridSearchCV(SVR(), param_grid = param_grid, cv = kfold, scoring = "neg_mean_squared_error",n_jobs = -1,verbose = 1)

SVMR.fit(X_train,y_df)
SVMR_best = SVMR.best_estimator_
print(SVMR.best_params_)

# cv = 20 

# * {'kernel': 'poly', 'C': 0.1, 'gamma': 'auto', 'degree': 2, 'epsilon': 0.03, 'coef0': 1.5} - 0.12514
# ** {'kernel': 'poly', 'C': 0.1, 'gamma': 'auto', 'degree': 2, 'epsilon': 0.03, 'coef0': 1.6} - 0.12428


# In[ ]:


print("SVM mean score:", rmsle_cv(SVMR_best).mean())
print("SVM std:", rmsle_cv(SVMR_best).std())


# In[ ]:


y_submission_4 = np.expm1(SVMR.predict(X_test))


# In[ ]:





# #### Random Forest

# In[ ]:


RFC = RandomForestRegressor(random_state = 1)

rf_param_grid = {"max_depth": [None],
              "max_features": [88],
              "min_samples_leaf": [1],
              "n_estimators" :[570]
                }

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv = kfold, scoring = "neg_mean_squared_error", n_jobs = -1, verbose = 1)
gsRFC.fit(X_train,y_df)
RFC_best = gsRFC.best_estimator_
print(gsRFC.best_params_)


# cv = 20 (Scaler)
# {'max_depth': None, 'min_samples_leaf': 1, 'max_features': 88, 'n_estimators': 600}
# {'max_depth': None, 'min_samples_leaf': 1, 'max_features': 88, 'n_estimators': 570} - 0.13778


# In[ ]:


print("Random Forest mean score:", rmsle_cv(RFC_best).mean())
print("Random Forest std:", rmsle_cv(RFC_best).std())


# In[ ]:


y_submission_5 = np.expm1(gsRFC.predict(X_test))


# In[ ]:





# #### XG Boost

# In[ ]:


XGB = XGBRegressor()

xg_param_grid = {
              'n_estimators' :[870],
              'learning_rate': [0.04],
              
              'max_depth': [3],
              'min_child_weight':[0.2],
              
              'gamma': [0],
                
              'subsample':[0.8],
              'colsample_bytree':[0.7]
    
              #'reg_alpha':[0.08,0.09,0.095,0.1,0.15,0.2],
              #'reg_lambda':[0,0.001,0.002]
              }
                
gsXGB = GridSearchCV(XGB,param_grid = xg_param_grid, cv=kfold, scoring="neg_mean_squared_error", n_jobs= -1, verbose = 1)
gsXGB.fit(X_train,y_df)
XGB_best = gsXGB.best_estimator_
print(gsXGB.best_params_)

# cv = 20
# {'min_child_weight': 0.5, 'learning_rate': 0.05, 'n_estimators': 850, 'max_depth': 3} - 0.12611
# {'min_child_weight': 0.2, 'learning_rate': 0.04, 'gamma': 0, 'n_estimators': 870, 'max_depth': 3}
# * {'max_depth': 3, 'subsample': 0.8, 'learning_rate': 0.04, 'gamma': 0, 'colsample_bytree': 0.7, 'min_child_weight': 0.2, 'n_estimators': 870} - 0.12287
# {'gamma': 0, 'min_child_weight': 0.1, 'learning_rate': 0.04, 'n_estimators': 885, 'max_depth': 3}

# {'reg_alpha': 0.1, 'reg_lambda': 0.001, 'n_estimators': 870, 'colsample_bytree': 0.7, 'subsample': 0.8, 'min_child_weight': 0.2, 'learning_rate': 0.04, 'gamma': 0, 'max_depth': 3} - 0.12531
# 


# In[ ]:


print("XG Boost mean score:", rmsle_cv(XGB_best).mean())
print("XG Boost std:", rmsle_cv(XGB_best).std())


# In[ ]:


y_submission_6 = np.expm1(gsXGB.predict(X_test))


# In[ ]:





# ### 5.3 Ensemble - Stacked Regression and GridSearch <a id='stacked_regression_and_gridsearch'></a>
# 
# Stacking is an ensemble learning technique to combine multiple regression models via a meta-regressor. In our case, we will use XG Boost as a meta-regressor and use the predictions of Lasso Regression, Elasticnet and XG Boost as trainig set of stacking. The reason why I chose the two models is that these are performing the best among many algorithms. Just be aware that it is not always that combination of only the best models performs the best. Selection of which models to choose for stacking is more like art rather science. Sometimes, some models that perform not well as a single model may work well on stacking so always do some experiments with many combinations. I tried many combinations but in my case, the best single models perform the best on stacking as well. One tip for the experiment that I have just mentioned is that it usually performs better as the models will be used for stacking have different characteristics or mechanisms. Below illustration is showing how stacking works.

# In[ ]:


print("source: https://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/")
Image(url= "https://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor_files/stacking_cv_regressor_overview.png")


# In[ ]:


XGB = XGBRegressor()

ELNET = ElasticNet(random_state = 1)
LCV=Lasso(random_state = 1)
SV = SVR()
KR = KernelRidge()
XG = XGBRegressor()
stack = StackingRegressor(regressors = [ELNET,LCV,XG],meta_regressor = XGB)

params = {       
              'meta-xgbregressor__n_estimators' : [740*2],#740
              'meta-xgbregressor__learning_rate': [0.01/2], #0.01
              'meta-xgbregressor__min_child_weight':[0],
              'meta-xgbregressor__gamma':[0.1],
              'meta-xgbregressor__max_depth': [2],
              'meta-xgbregressor__subsample':[0.65],
              'meta-xgbregressor__colsample_bytree':[0.4],
              'meta-xgbregressor__reg_alpha':[0],
              'meta-xgbregressor__reg_lambda':[1],
              
              'lasso__alpha':[0.00244736842105],
              'elasticnet__alpha':[0.0276315789474],
              'elasticnet__l1_ratio':[0.09],
              'xgbregressor__min_child_weight':[0.2],
              'xgbregressor__n_estimators' : [870],
              'xgbregressor__learning_rate': [0.04],
              'xgbregressor__gamma':[0],
              'xgbregressor__max_depth': [3],
              'xgbregressor__subsample':[0.8],
              'xgbregressor__colsample_bytree':[0.7]
    
              #'kernelridge__alpha':[0.93],
              #'kernelridge__coef0':[1.5],
              #'kernelridge__degree':[3],
              #'kernelridge__gamma':[0.001],
              #'kernelridge__kernel':['polynomial'],
              #'kernelridge__kernel_params':[None],
              
              #'svr__coef0':[1.6],
              #'svr__kernel':['poly'],
              #'svr__epsilon':[0.03],
              #'svr__gamma': ['auto'],
              #'svr__degree': [2],
              #'svr__C':[0.1]
        }

grid = GridSearchCV(estimator = stack, param_grid=params,cv=kfold,refit=True, verbose=1,n_jobs=1,scoring="neg_mean_squared_error")
grid.fit(X_train, y_df)
grid_best = grid.best_estimator_
print(grid_best)

#StackingRegressor(meta_regressor=XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
#learning_rate=0.01, max_delta_step=0, max_depth=3,
#min_child_weight=0.5, missing=None, n_estimators=770, nthread=-1,
#objective='reg:linear', reg_alpha=0, reg_lambda=1,
#scale_pos_weight=1, seed=0, silent=True, subsample=1) - 0.12965

# StackingRegressor(meta_regressor=XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
# gamma=0, learning_rate=0.02, max_delta_step=0, max_depth=1,
# min_child_weight=0.3, missing=None, n_estimators=760, nthread=-1,
# objective='reg:linear', reg_alpha=0, reg_lambda=1,
# scale_pos_weight=1, seed=0, silent=True, subsample=0.3) - 0.12546

#StackingRegressor(meta_regressor=XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.5,
#gamma=0, learning_rate=0.02, max_delta_step=0, max_depth=1,
#min_child_weight=0.2, missing=None, n_estimators=760, nthread=-1,
#objective='reg:linear', reg_alpha=0, reg_lambda=1,
#scale_pos_weight=1, seed=0, silent=True, subsample=0.2) - 0.12493

#StackingRegressor(meta_regressor=XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.4,
#gamma=0.1, learning_rate=0.01, max_delta_step=0, max_depth=2,
#min_child_weight=0, missing=None, n_estimators=740, nthread=-1,
#objective='reg:linear', reg_alpha=0, reg_lambda=1,
#scale_pos_weight=1, seed=0, silent=True, subsample=0.65) - 0.12027

#StackingRegressor(meta_regressor=XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.4,
#gamma=0.1, learning_rate=0.005, max_delta_step=0, max_depth=2,
#min_child_weight=0, missing=None, n_estimators=1480, nthread=-1,
#objective='reg:linear', reg_alpha=0, reg_lambda=1,
#scale_pos_weight=1, seed=0, silent=True, subsample=0.65) - 0.12026


# In[ ]:


print("Stacking mean score:", rmsle_cv(grid_best).mean())
print("Stacking std:", rmsle_cv(grid_best).std())


# In[ ]:


y_submission_st = np.expm1(grid.predict(X_test))


# The best result comes from the combination of Lasso Regression, ElasticNet and XG Boost, which scores 0.12026 and best score so far.

# In[ ]:





# ### 5.4 Ensemble - Averaging 
# <a id='avg'></a>
# 
# The result of stacking outperforms all the single models'. One more time, combining the stacking result with best single models would be able to boost our accuracy even more; however, this time, instead of stacking, we can use different type of ensemble, averaging. Simply, averaging is adding all the results predicted by each model and dividing by the number of models. First time, I expected weighted averaging would make a better prediction since the stacking result outperforms other models; however, on the contrary to my expectation, just simple averaing method performs the best.

# In[ ]:


y_submission_avg = (y_submission_6 + y_submission_2 + y_submission_st)/3

# W: 0.11960 (y_submission_6 + y_submission_2)/2
# WW: 0.11948 (y_submission_6 + y_submission_2 + y_submission_st)/3


# In[ ]:


#y_submission_weight = (y_submission_st *0.3340) + (y_submission_2 * 0.3331) + (y_submission_6 *0.3329) - 11.952
# y_submission_weight = (y_submission_st *0.334) + (y_submission_2 * 0.3331) + (y_submission_6 *0.3329) - 11.952


# The final result is 0.11948 which is placed in top 17% in the competition.

# In[ ]:





# ## 6. Submission
# <a id='submission'></a>

# In[ ]:


my_submission = pd.DataFrame()
my_submission['Id'] = test_id
my_submission['SalePrice'] = y_submission_avg
my_submission.to_csv('submission47.csv',index=False)

