#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# ### Welcome!
# Hey All, this is my first full end-to-end Kaggle project. We all will learn about automated machine learning with help TPOT library along with the basic framework for any data science competition. My goal is to learn and contribute to the data science community. 
# 
# You all must be wondering....**What's in it for me?**
# * Understanding the problem
# * EDA and data visualization
# * Feature Engineering
# * Modelling 
# 
# 
# I have referred to some of the best kernel, to name a few
# * <a href="https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-pythonComprehensive"> Data Exploration with Python</a>
# * <a href="https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard">
# Stacked Regressions to predict House Prices</a>

# ## Understanding the problem

# #### Competition Description
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. <br>
# 
# Approach : For this problem we will first analyze all the features affecting the price of the house and thereby building a predictive model to predict house price (price is a number from some defined range, so it will be regression task). **Let's begin**

# ### Importing required libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, skew, boxcox_normmax


# ### Reading the housing price data

# In[ ]:


hp_train = pd.read_csv("../input/train.csv")
hp_test = pd.read_csv("../input/test.csv")


# In[ ]:


hp_train.head()


# In[ ]:


print("Train set size:", hp_train.shape)
print("Test set size:", hp_test.shape)


# In[ ]:


train_ID = hp_train['Id']
test_ID = hp_test['Id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
hp_train.drop(['Id'], axis=1, inplace=True)
hp_test.drop(['Id'], axis=1, inplace=True)


# > ## Performing EDA to know more about data followed by preprocessing and data preparation

# There is famous quote in data science "*Quality of your inputs decide the quality of your output*", so let's get started!!

# ### Outliers
# A value that "lies outside" (is much smaller or larger than) most of the other values in a set of data.<br>
# There are numerous unfavourable impacts of outliers in the data set:
# * It increases the error variance and reduces the power of statistical tests
# * If the outliers are non-randomly distributed, they can decrease normality
# * They can bias or influence estimates that may be of substantive interest
# * They can also impact the basic assumption of Regression and other statistical model assumptions.
# 

# In[ ]:


col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']
outlier = [4500, 3000, 2500, 2000, 55000]
for i, c in zip(range(5), col_name):
    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.scatter(np.abs(hp_train[hp_train[c] < outlier[i]][c]), np.array(hp_train[hp_train[c] < outlier[i]]['SalePrice']), c='b')
    plt.scatter(np.abs(hp_train[hp_train[c] >= outlier[i]][c]), np.array(hp_train[hp_train[c] >= outlier[i]]['SalePrice']), c='r')
    plt.title('Before removing outliers for '+c)
    plt.xlabel(c)
    plt.ylabel('SalePrice')
    
    
    plt.subplot(1,2,2)
    plt.scatter(np.abs(hp_train[hp_train[c] < outlier[i]][c]), np.array(hp_train[hp_train[c] < outlier[i]]['SalePrice']), c='b')
    plt.title('After removing outliers for '+c)
    plt.xlabel(c)
    plt.ylabel('SalePrice')
    plt.show()


# In[ ]:


# removing outliers
print(hp_train.shape)
hp_train = hp_train[hp_train['GrLivArea'] < 4500]
hp_train = hp_train[hp_train['LotArea'] < 550000]
hp_train = hp_train[hp_train['TotalBsmtSF'] < 3000]
hp_train = hp_train[hp_train['1stFlrSF'] < 2500]
hp_train = hp_train[hp_train['BsmtFinSF1'] < 2000]


# ### Our Target variable is **SalePrice**

# In[ ]:


#Describing SalePrice
hp_train.SalePrice.describe()


# In[ ]:


#Understanding the distribution of SalePrice
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(hp_train['SalePrice'])
plt.title('Understanding the distribution of SalePrice')

plt.subplot(1,2,2)
stats.probplot((hp_train['SalePrice']), plot=plt)
plt.show()


# In[ ]:


print("Skewness: %f" % hp_train['SalePrice'].skew())
print("Kurtosis: %f" % hp_train['SalePrice'].kurt())


# Observations from the distribution
# * The Sales price distribution is right skewed, which is not normal. Hence we need to transform into a normal distribution
# * Also from Quantile-Quantile plot it is evident that the distribution is not normal
# * If we observe distribution properly we see that it has peak, which is also evident from kurtosis value
# 

# #### Transforming distribution of target variable into normal distribution

# Common transformations of right skewed data includes square root, cube root, and log transformations.

# In[ ]:


fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(np.sqrt(hp_train['SalePrice']))
plt.title('Distribution of SalePrice after square root transformation')

plt.subplot(1,2,2)
stats.probplot(np.sqrt(hp_train['SalePrice']), plot=plt)
plt.title('Square root transformation')
plt.show()

fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(np.cbrt(hp_train['SalePrice']))
plt.title('Distribution of SalePrice after cube root transformation')

plt.subplot(1,2,2)
stats.probplot(np.cbrt(hp_train['SalePrice']), plot=plt)
plt.title('Cube root transformation')
plt.show()

fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(np.log1p(hp_train['SalePrice']))
plt.title('Distribution of SalePrice after log transformation')

plt.subplot(1,2,2)
stats.probplot(np.log1p(hp_train['SalePrice']), plot=plt)
plt.title('Log transformation')
plt.show()


# In[ ]:


print("After log transformation")
hp_train['SalePrice'] = np.log1p(hp_train['SalePrice']) 
print("Skewness: %f" % (hp_train['SalePrice'].skew()))
print("Kurtosis: %f" % (hp_train['SalePrice'].kurt()))


# We will go forward with **log transformation**, as it is good fit compared to other transformation

# In[ ]:


hp_train.SalePrice.describe()


# ### Relationships with SalePrice
# We studied the target variable, now we will be analyzing the features which influences the target variable.

# In[ ]:


#Correlation map to see how features are correlated with SalePrice
corrmat = hp_train.corr()

# select top 10 highly correlated variables with SalePrice
num = 10
col = corrmat.nlargest(num, 'SalePrice')['SalePrice'].index
coeff = np.corrcoef(hp_train[col].values.T)

mask = np.zeros_like(coeff, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig = plt.figure(figsize=(15,10))
sns.heatmap(coeff, vmin = -1, annot = True, mask = mask, square=True, xticklabels = col.values, yticklabels = col.values);


# In[ ]:


#Categorical Variables
fig, axes = plt.subplots(ncols=4, nrows=4, 
                         figsize=(4 * 4, 4 * 4), sharey=True)

axes = np.ravel(axes)

cols = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',
        'BsmtCond','GarageQual','GarageCond', 'MSSubClass','MSZoning',
        'Neighborhood','BldgType','HouseStyle','Heating','Electrical','SaleType']

for i, c in zip(np.arange(len(axes)), cols):
    ax = sns.boxplot(x=c, y='SalePrice', data=hp_train, ax=axes[i], palette="Set2")
    ax.set_title(c)
    ax.set_xlabel("")


# * Based on the correlation matrix, we can see that the features related with quality (OverallQual,FullBath, YearBuilt, YearRemodAdd) and the size (GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF) influences the sale price, which might impact our predictions
# * And based on the correlation plot for the categorical features we can conclude that some variables have influence on the SalePrice and the OverAllQuality seems to have the highest influence

# ### Concatenation of train and test datasets

# In[ ]:


ntrain = hp_train.shape[0]
ntest = hp_test.shape[0]
y_train = hp_train.SalePrice.values
df_all = pd.concat((hp_train, hp_test), sort=False).reset_index(drop=True)
df_all.drop(['SalePrice'], axis=1, inplace=True)
print("df_all size after concatenation of train and test data is : {}".format(df_all.shape))


# ## Feature Engineering

# In[ ]:


df_all.info()


# In[ ]:


missing_data = pd.DataFrame({'total_missing': df_all.isnull().sum(), 'perc_missing': (df_all.isnull().sum()/len(df_all))*100})
len(missing_data[missing_data.total_missing>0])


# In[ ]:


fig = plt.figure(figsize=(15,10))
missing_data[missing_data.total_missing>0].sort_values(by='perc_missing')['perc_missing'].plot(kind='barh')
plt.xlabel('Percentage of missing values')
plt.ylabel('Features')
plt.title('Percentage of missing values for different features')
plt.show()


# We can extend our observations on missing data and the datatypes here:
# 
# * Out of 79 columns, 34 columns have incomplete data, which we need to treat
# * Quite a lot of data seems to be missing in PoolQC, MiscFeature, Alley and Fence, as most houses won't have pool, fence and alley
# * Alot of the columns have strings (object datatype), which needs to be parsed into the category datatype 

# ### Treating Missing values

# First, lets understand each features with missing values<br>
# And, what NA in the following features means - 
# 
# | Features | NA here means | Treament (Categorical / Numeric) |
# | --- | --- | --- |
# | PoolQC | No pool | None / 0 |  
# | MiscFeature | No misc feature | None / 0 |
# | Alley | No alley access | None / 0 |
# | Fence | No fence | None / 0 |
# | FireplaceQu | No fireplace | None / 0 |
# | Garage-related features | No garage | None / 0 |
# | Basement-related features | No basement | None / 0 |
# | Masonay-related features | No masonry veneer | None / 0 |
# | LotFrontage | Information not available | KNN |
# | Electrical | Information not available | Mode |
# | MSZoning | Information not available | Mode |
# | Utilities | Information not available | Mode |
# | SaleType | Information not available | Mode |
# | KitchenQual | Information not available | Mode |
# | Exterior - related | Information not available | Mode |
# | Functional | Typical | Typ |
# 

# **Keep calm and lets start treating missing values**

# In[ ]:


#Creating a list for features with No amenities
no_amen_cat = ['PoolQC','MiscFeature','Alley','Fence','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'FireplaceQu']
no_amen_num = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']


# In[ ]:


df_all1 = df_all.copy() #Creating a copy of original dataframe


# In[ ]:


for col in no_amen_cat:
    df_all1[col] = df_all1[col].fillna('None')


# In[ ]:


for col in no_amen_num:
    df_all1[col] = df_all1[col].fillna(0)


# In[ ]:


mode_replace = ['Electrical', 'MSZoning', 'Utilities', 'SaleType', 'KitchenQual', 'Exterior2nd', 'Exterior1st']
for col in mode_replace:
    df_all1[col] = df_all1[col].fillna(df_all1[col].mode()[0]) 


# In[ ]:


df_all1["Functional"] = df_all1["Functional"].fillna("Typ")


# In[ ]:


df_all1.isna().sum()[df_all1.isna().sum()>0].sort_values(ascending=False)


# So, we treated 33 out of 34 features with missing value. **Hold on for one last feature**<br>
# We will be filling all the missing values for LotFrontage using median value from its neighbours, as area of each street connected to the house property most likely to have a similar area to other houses in its neighborhood

# In[ ]:


df_all1["LotFrontage"] = df_all1.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[ ]:


df_all1.isna().sum()[df_all1.isna().sum()>0].sort_values(ascending=False)


# In[ ]:


df_all2 = df_all1.copy() #Creating a copy of original dataframe


# **Kudos!! We did it**

# Well it is no surprise that our task now is to somehow extract the information out of the categorical variables, but choosing the right encoder is also important<br>
# Following is the article on same topic
# https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b<br>
# Label Encoding refers to converting the labels into numeric form for ordinal features so as to convert it into the machine-readable form by retaining the order

# In[ ]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df_all2[c].values)) 
    df_all2[c] = lbl.transform(list(df_all2[c].values))

# shape        
print('Shape : {}'.format(df_all2.shape))


# Lets now look at the skewness of the features. **Now why is this important?** <br>
# Parameter estimation is based on the minimization of squared error. observations in skewed data will make a disproportionate effect on the parameter estimates. Hence we need to transform highly skewed features into normal distribution 

# In[ ]:


numeric_feats = df_all2.dtypes[df_all2.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df_all2[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In the beginning of the notebook, we already transformed right skewed target feature (SalePrice) into normal distribution, but there we compared multiple transformation and finally went with log transformation. Here instead we will be using Box Cox transform, which actually helped to increase the accuracy of our prediction<br>

# In[ ]:


skewness = skewness[abs(skewness) > 0.5]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.45
for feat in skewed_features:
    df_all2[feat] = boxcox1p(df_all2[feat], lam)


# In[ ]:


#Dropping dominating features with more than 95% same values
df_all2 = df_all2.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis = 1)


# For categorical features without any orders we will be using get_dummies to convert into numerical form

# In[ ]:


df_all2 = pd.get_dummies(df_all2).reset_index(drop=True)
print(df_all2.shape)


# In[ ]:


#Adding new features from existing features
df_all2['TotalSF']=df_all2['TotalBsmtSF'] + df_all2['1stFlrSF'] + df_all2['2ndFlrSF']

df_all2['Total_sqr_footage'] = (df_all2['BsmtFinSF1'] + df_all2['BsmtFinSF2'] +
                                 df_all2['1stFlrSF'] + df_all2['2ndFlrSF'])

df_all2['Total_Bathrooms'] = (df_all2['FullBath'] + (0.5 * df_all2['HalfBath']) +
                               df_all2['BsmtFullBath'] + (0.5 * df_all2['BsmtHalfBath']))

df_all2['Total_porch_sf'] = (df_all2['OpenPorchSF'] + df_all2['3SsnPorch'] +
                              df_all2['EnclosedPorch'] + df_all2['ScreenPorch'] +
                              df_all2['WoodDeckSF'])


# In[ ]:


#Creating a binary features from existing features
df_all2['haspool'] = df_all2['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df_all2['has2ndfloor'] = df_all2['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df_all2['hasgarage'] = df_all2['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_all2['hasbsmt'] = df_all2['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df_all2['hasfireplace'] = df_all2['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:


df_all2.shape


# Train Test Split

# In[ ]:


train = df_all2[:ntrain]
test = df_all2[ntrain:]


# ## Modeling

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from  sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LassoLarsCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.builtins import ZeroCount
from sklearn.decomposition import PCA
from imblearn.pipeline import make_pipeline


# In[ ]:


#Defining score as we will need to score for multiple models
def score(model, train_x, train_y):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv=5, scoring="neg_mean_squared_error"))
    print(score.mean(), score.std())


# For our model selection we will be using TPOT, which is *An Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming*. Go through <a href="https://github.com/EpistasisLab/tpot"> TPOT</a> for further information
# 

# In[ ]:


from tpot import TPOTRegressor
tpot = TPOTRegressor(generations=1,verbosity=2,scoring='neg_mean_squared_error')


# In[ ]:


tpot.fit(train.values, y_train)


# In[ ]:


#top performer
tpot.score(train.values, y_train)


# In[ ]:


models_tested = pd.DataFrame(tpot.evaluated_individuals_).transpose()


# In[ ]:


#We will using top 5 models for our final submission
models_tested.sort_values(['internal_cv_score'], ascending=False).head(10)


# > Selecting best 5 models from TPOT

# In[ ]:


exported_pipeline1 = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=0.7000000000000001, tol=0.001, cv=5)
)

exported_pipeline1.fit(train.values, y_train)


# In[ ]:


score(exported_pipeline1, train.values, y_train)
exported_pipeline_pred1 = exported_pipeline1.predict(train.values)
score(exported_pipeline1, train.values, exported_pipeline_pred1)


# In[ ]:


exported_pipeline2 = make_pipeline(
    MinMaxScaler(),
    ElasticNetCV(l1_ratio=0.7000000000000001, tol=0.01, cv=5)
)

exported_pipeline2.fit(train.values, y_train)


# In[ ]:


score(exported_pipeline2, train.values, y_train)
exported_pipeline_pred2 = exported_pipeline2.predict(train)
score(exported_pipeline2, train.values, exported_pipeline_pred2)


# In[ ]:


exported_pipeline3 = make_pipeline(
    ZeroCount(),
    PCA(iterated_power=4, svd_solver='randomized'),
    RidgeCV()
)

exported_pipeline3.fit(train.values, y_train)


# In[ ]:


score(exported_pipeline3, train.values, y_train)
exported_pipeline_pred3 = exported_pipeline3.predict(train)
score(exported_pipeline3, train.values, exported_pipeline_pred3)


# In[ ]:


exported_pipeline4 = make_pipeline(
    LassoLarsCV(normalize=True, max_iter=60, cv=5)
)

exported_pipeline4.fit(train.values, y_train)


# In[ ]:


score(exported_pipeline4, train.values, y_train)
exported_pipeline_pred4 = exported_pipeline4.predict(train)
score(exported_pipeline4, train.values, exported_pipeline_pred4)


# In[ ]:


exported_pipeline5 = make_pipeline(
    RidgeCV()
)

exported_pipeline5.fit(train.values, y_train)


# In[ ]:


score(exported_pipeline5, train.values, y_train)
exported_pipeline_pred5 = exported_pipeline5.predict(train)
score(exported_pipeline5, train.values, exported_pipeline_pred5)


# In[ ]:


#Checking score after applying equal wightage to all models
np.sqrt(mean_squared_error(y_train,(exported_pipeline_pred1*0.2 + exported_pipeline_pred2*0.2 +
               exported_pipeline_pred3*0.2 + exported_pipeline_pred4*0.2 + exported_pipeline_pred5*0.2)))


# In[ ]:


#Here we will be deciding weightage of the models based on random selection which gives the least loss
best_value = []
min_value=1
for i in range(10000):
    random = np.random.dirichlet(np.ones(5),size=1)
    best_value.append(np.sqrt(mean_squared_error(y_train,(exported_pipeline_pred1*random[0][0] + exported_pipeline_pred2*random[0][1] +
               exported_pipeline_pred3*random[0][2] + exported_pipeline_pred4*random[0][3] + exported_pipeline_pred5*random[0][4]))))
     
    if(np.min(best_value) < min_value):
        min_value = np.min(best_value)
        min_array = random


# In[ ]:


np.min(best_value)


# In[ ]:


min_array


# In[ ]:


def blend_models(sub):
    return (exported_pipeline1.predict(sub)*min_array[0][0] + 
            exported_pipeline2.predict(sub)*min_array[0][1] + 
            exported_pipeline3.predict(sub)*min_array[0][2] + 
            exported_pipeline4.predict(sub)*min_array[0][3] + 
            exported_pipeline4.predict(sub)*min_array[0][4])


# In[ ]:


print('Predict submission')
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = np.floor(np.expm1(blend_models(test)))


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)


# **Next Steps:**
# * We will train TPOT model for further generations
# * We can further use deep learning to solve this, as it is best choice for non-linear features 

# 1. I hope this kernel helps you all!
