#!/usr/bin/env python
# coding: utf-8

# # This is a kernel for house price:advanced regression techniques dataset

# ## Imports

# In[ ]:


#performing imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy import stats
import warnings
import missingno as msno
from scipy.stats import boxcox,skew,norm
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#reading the trainset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#Taking a look at the columns and shape of the trainset
print("Columns of trainset: ",train.columns)
print("Shape of the trainset: ",train.shape)


# In[ ]:


#printing the head of the trainset
train.head()


# In[ ]:


#listing the numeric features
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns


# In[ ]:


#listing the categorical features
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns


# In[ ]:


#the ID column is only for indexing and won't help much in training the ML model.So it's safe to delete it

train_ID = train['Id']
test_ID = test['Id']

train = train.drop("Id",axis=1,inplace=False)
test = test.drop("Id",axis=1,inplace=False)

#printing the shapes of train set and test set just to verify that the ID column has been dropped
print("Trainset shape ",train.shape)
print("Testset shape ",test.shape)


# ## Data Analysis

# First of all, just to be informed, SalePrice is our target variable. Let's see it's stats using the describe function

# In[ ]:


train['SalePrice'].describe()


# In[ ]:


#visualising a histogram of the target variable to see it's distribution
sns.distplot(train['SalePrice'])


# As you can see, there are some things noteworthy:<br>
# -> There's a deviation from normal distribution<br>
# -> Having positive skewness<br>
# -> Having peakedness<br>

# In[ ]:


#calculating skewness and kutrosis for SalePrice
print("Skewness: ",train['SalePrice'].skew())
print("Kurtosis: ",train['SalePrice'].kurt())


# For an ideal distribution, skewness and kurtosis both should be zero

# #### Creating correlation matrices both general and zoomed style

# In[ ]:


#creating a correlation matrix
corr_mat = train.corr()
print(corr_mat['SalePrice'].sort_values(ascending=False),'\n')


# The maximum value of the correlation aside from itself is 0.79, so let's keep the vmax for the sns heatmap be 0.8 to have a more customised heatmap

# In[ ]:


f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corr_mat,square=True,vmax=0.8)


# Some noteworthy things:<br>
# ->'TotalBsmtSF' and '1stFlrSF' are highly correlated<br>
# ->'GarageCars' and 'GarageArea' are highly correlated<br>
# ->The Target Variable SalePrice is likely to be correlated to 'OverallQual','GrLivArea','TotalBsmtSF' and 'GarageCars'

# #### ZoomedHeatMap

# Let's create a zoomed heat map using the top 12 highly correlated features with SalePrice

# In[ ]:


k= 11
cols = corr_mat.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)


# Some notes:<br>
# -> OverallQual,GrLiveArea,GarageCars, GarageArea and TotalBsmtSF are strongly correlated to SalePrice<br>
# -> GarageCars and GarageArea are so correlated to each other and their correlation matrix shows they can be replaced with each other. GarageCars has more correlation to SalePrice than GarageArea and GarageArea should be dropped<br>
# ->TotalBsmtSF and 1stFlrSF can also be considered as twins<br>
# ->TotRmsAbvGrd and GrLivArea are twins as well<br>

# #### Plotting Correlations of Sale Price with some of the strong correlating variables

# #### Relationship with Numerical Variables

# The limit on y can be set at 800000 as we have seen that the max value on y is 755000

# In[ ]:


#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))


# In[ ]:


#scatter plot garagecars/saleprice. this is not a numerical variable but a dummy variable
var = 'GarageCars'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))


# In[ ]:


#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))


# #### Relationship with Categorical Features

# In[ ]:


#bos plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
f, ax = plt.subplots(figsize=(12,9))
fig = sns.boxplot(x=var, y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)


# In[ ]:


#box plot yearbuilt/saleprice
var = 'YearBuilt'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
f,ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90)


# These are the plots with the highly correlated variables.Some noteworthy points can be:<br>
# ->GrLivArea and SalePrice are having a linear relationship<br>
# ->The GarageCars is likely to have outliers on its value 3<br>
# ->TotalBsmtSF and SalePrice is likely to have an exponential relationship<br>
# ->OverallQual and SalePrice are highly related<br>
# ->The Price is high as the year passes, which can be seen from the yearbuilt graph<br>

# #### Scatterplots between SalePrice and all other variables

# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(train[cols],size=3)
plt.show()


# ### Dealing with Outliers

# Outliers are extreme observations in our data which can affect the accuracy of the trained model.If such observations are too less, it's safe to delete them.

# Lets plot some subplots which can be used to determine some outliers

# #### GrLivArea

# In[ ]:


fig, ax = plt.subplots(figsize=(16,9))
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers. Therefore, we can safely delete them.

# In[ ]:


#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index) #creating a virtual box to corner these
                                                                                        #outliers

#Check the graphic again
fig, ax = plt.subplots(figsize=(16,9))
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# #### TotalBsmtSF

# In[ ]:


fig, ax = plt.subplots(figsize=(16,9))
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# #### Note:

# It's tempting to delete the 3 values where GrLivArea>3000. But i think that it won't affect the regression function that much. So I'm keeping them as they are. Also one thing to note during outlier deletion is that it's a trade off. The records are being deleted and so are valuable observations for that record. Outlier deletion should only be done if it seems that the recorded observation is truly an outlier and not just an observation with slight more variance

# #### Normality

# We will be attending to:<br>
# <b>Histogram: </b> Kurtosis and skewness<br>
# <b>Normal probability plot: </b>Data Distribution should closely follow the diagnol that represents the normal distribution

# <b>1.SalePrice</b>

# In[ ]:


#histogram and normal probability plot
sns.distplot(train['SalePrice'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['SalePrice'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('SalePrice')

fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()


# Some notes:<br>
# ->The distribution is not good. Positive skewness and 'peakedness'.The probability plot doesn't follow the normal distribution diagnol<br>
# 
# ->Solution: When there is positive skewness, log transformations are really helpful

# In[ ]:


#applying log transformation
train['SalePrice'] = np.log(train['SalePrice']+1)


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(train['SalePrice'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['SalePrice'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('SalePrice')

fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()


# The SalePrice variable is dealt with

# <b>2.GrLivArea</b>

# In[ ]:


#histogram and normal probability plot
sns.distplot(train['GrLivArea'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['GrLivArea'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('GrLivArea')

fig = plt.figure()
res = stats.probplot(train['GrLivArea'],plot=plt)
plt.show()


# There is a positive skewness in the distribution. We already know the solution

# In[ ]:


#applying log transformation
train['GrLivArea'] = np.log(train['GrLivArea']+1)


# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(train['GrLivArea'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['GrLivArea'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('GrLivArea')

fig = plt.figure()
res = stats.probplot(train['GrLivArea'],plot=plt)
plt.show()


# <b>3.TotalBsmtSF</b>

# In[ ]:


#histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['TotalBsmtSF'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('TotalBsmtSF')

fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'],plot=plt)
plt.show()


# Now this is a tricky one to deal with. Why? because:<br>
# ->Positive Skewness<br>
# ->If you observe the distribution carefully, there are many 0 values. so? whats the problem right?<br>
# ->No, because Log transformations can't happen on 0 values.<br>
# 
# ->Solution: As per my research, people usually use log(x+c), but as you can see below, it doesn't exactly solve the problem.

# In[ ]:


#transformed histogram and normal probability plot
sns.distplot(np.log(train['TotalBsmtSF']+1),fit=norm)
fig = plt.figure()
res = stats.probplot(np.log(train['TotalBsmtSF']+1),plot=plt)

#Note:This doesn't seem well


# Another solution is doing a boxcox transformation. But the boxcox transformation requires the data to be strictly positive (i.e >0). Hence combining the above approach and then doing the boxcox transformation.<br>
# <b>Note:</b> I'm not entirely sure about this being the best solution to deal with this problem

# In[ ]:


train['TotalBsmtSF'],maxlog = boxcox(train['TotalBsmtSF']+1)

#histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['TotalBsmtSF'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('TotalBsmtSF')

fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'],plot=plt)
plt.show()


# ### Dealing with Missing Values

# In[ ]:


#working with the missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending=False)
missing_vals = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_vals.head(30)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
data_na = missing_vals[:20]
sns.barplot(x=data_na.index, y=data_na.Percent)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# I'm gonna fill up the missing values here. Although it should be noted that variables such as PoolQC and MiscFeature don't add much value here and can be probably removed as well.

# PoolQC

# In[ ]:


train['PoolQC'] = train['PoolQC'].fillna('None')
test['PoolQC'] = test['PoolQC'].fillna('None')


# MiscFeature

# In[ ]:


train['MiscFeature'] = train['MiscFeature'].fillna('None')
test['MiscFeature'] = test['MiscFeature'].fillna('None')


# Alley

# In[ ]:


train['Alley'] = train['Alley'].fillna('None')
test['Alley'] = test['Alley'].fillna('None')


# Fence

# In[ ]:


train['Fence'] = train['Fence'].fillna('None')
test['Fence'] = train['Fence'].fillna('None')


# FireplaceQu

# In[ ]:


train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')


# Lot Frontage

# In[ ]:


#filling the lot frontage with the median value. A good trick is to use the median value of the neighbourhood rather than
#lotfrontage, so that it can be realistic in regards to other parameters as well.
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#A thing to note is why train and test be treated like the same but kept different. This is because the test data doesn't leak
#in the training set and your model doesn't 'know' your test data before testing


# GarageType, GarageFinish, GarageQual and GarageCond

# In[ ]:


for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')


# I'm filling all the garage and basement missing variables with 0 as no garage=no cars and also i think missing values for no basement are likely to be 0

# In[ ]:


for col in ('GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)


# For the categorical variables MasVnrArea and MasVnrType. NA means most likely no mason as None is much on MasVnrType.For the area we can fill 0

# In[ ]:


train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['MasVnrType'] = train['MasVnrType'].fillna('None')


# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2. Filling them with None

# In[ ]:


for col in('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')


# Electrical has mostly 'SBrkr'. we can use that to fill it up

# In[ ]:


train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])


# In[ ]:


train.shape


# In[ ]:


test.shape


# ### Data Transformation

# In[ ]:


#Transforming some numerical variables that are really categorical

#MSSubClass = The building class
train['MSSubClass'] = train['MSSubClass'].apply(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)

#Changing OverallQual into categorical variable
train['OverallQual'] = train['OverallQual'].astype(str)
test['OverallQual'] = test['OverallQual'].astype(str)

#Changing OverallCond into categorical variable
train['OverallCond'] = train['OverallCond'].astype(str)
test['OverallCond'] = test['OverallCond'].astype(str)

#Changing year sold and month sold into categorical features
train['YrSold'] = train['YrSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)


# In[ ]:


#Label Encoding some categorical variables so that their information can be used
categorical_features = train.select_dtypes(include=[np.object])
cols = list(categorical_features.columns)

#process all the columns, apply label encoding
for c in cols:
    lbl_train = LabelEncoder()
    lbl_train.fit(list(train[c].values))
    train[c] = lbl_train.transform(list(train[c].values))
    
    lbl_test = LabelEncoder()
    lbl_test.fit(list(test[c].values))
    test[c] = lbl_test.transform(list(test[c].values))    


# As such there exists skewness in many variables. They can be eliminated as well

# In[ ]:


numeric_feats = train.dtypes[train.dtypes!="object"].index

#check skew of all features
skewed_feats = train[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
print("Skew Features")
skewness = pd.DataFrame({'Skew' : skewed_feats})
print(skewness)


# To reduce the skewness here, I'm going to use the BoxCox transformation. But first of all, a threshold to check whether an observation is skewed or not has to be made. For the same, i'm assuming 0.65 as the threshold as I think it covers all the major variables fully and it would be sufficient for a proper training to the dataset

# In[ ]:


skewness = skewness[abs(skewness.Skew) > 0.65]
print("Features that can be skewed in the dataset are {0}".format(skewness.shape[0]))

skewed_features = skewness.index
for feat in skewed_features:
    train[feat] = np.log(train[feat]+1)
    #print("Lambda for maxlog for {0} is {1}. ".format(feat,maxlog))


# ### Getting dummy encoding features

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train,test)).reset_index(drop=True)
all_data.drop(['SalePrice'],axis=1,inplace=True)
print(all_data.shape)


# In[ ]:


all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head()


# ## Modelling

# In[ ]:


#Import libraries
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# ### Define a cross-validation strategy

# I'm going to use cross_val_score of sklearn. This function doesn't have a shuffle attribute, one code of line can be added in order to shuffle the dataset prior to cross-validation

# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,train.values,y_train,scoring='neg_mean_squared_error',cv=kf))
    return(rmse)


# ### Base Models

# 1.LASSO Regression

# It's sensitive to outliers. So RobustScaler() method can be used on the make_pipeline()

# In[ ]:


lasso_reg = make_pipeline(RobustScaler(),Lasso(random_state=42))


# 2.Elastic Net Regression

# In[ ]:


enet_reg = make_pipeline(RobustScaler(),ElasticNet(random_state=42))


# 3.Kernel Ridge Regression

# In[ ]:


krr_reg = KernelRidge(kernel="polynomial") #using krr instead of SVR because it's much faster on medium sized datasets


# 4.Gradient Boosting Regression

# huber loss makes it robust to outliers

# In[ ]:


gboost_reg = GradientBoostingRegressor(loss='huber',random_state=42)


# 5.XGBoost

# In[ ]:


xgb_reg = xgb.XGBRegressor(random_state=42)


# ### Base model scores

# In[ ]:


score = rmsle_cv(lasso_reg)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(enet_reg)
print("\nElastic Net score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(xgb_reg)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


xgb_reg.fit(train,y_train)
xgb_train_pred = xgb_reg.predict(train)
xgb_pred = np.expm1(xgb_reg.predict(test))
print(rmsle(y_train,xgb_train_pred))


# ### Submission

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = xgb_pred
sub.to_csv('submission.csv',index=False)


# In[ ]:


xgb_pred.shape


# In[ ]:




