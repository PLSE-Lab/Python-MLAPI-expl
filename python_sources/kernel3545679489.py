#!/usr/bin/env python
# coding: utf-8

# ### Team Name: 
# FAM
# 

# ### Team Members:
# 
# Asmaa Alrefae, Faisal Al shuraym, Mansour Aljuaid

# ## Problem Statement 
# 
# Predict the sales price for each house. For each Id in the test set, predict the value of the SalePrice variable. 

# The Kaggle Score for this Model is 0.122

# In[ ]:


#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


# In[ ]:


#Now let's import and put the train and test datasets in  pandas dataframe
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


##display the first five rows of the train dataset.
train.head(5)


# In[ ]:


##display the first five rows of the test dataset.
test.head(5)


# In[ ]:


#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# In[ ]:


#SalePrice is the variable we need to predict. So let's do some analysis on this variable first.
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


#The target variable is right skewed. As (linear) models love normally distributed data , 
#we need to transform this variable and make it more normally distributed.
#Log-transformation of the target variable
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# ## Data Processing 

# In[ ]:


#Outliers
#Documentation for the Ames Housing Data indicates that there are outliers present in the training data
train.describe().transpose()


# In[ ]:


#Let's explore these outliers

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


#We can see at the bottom right two with extremely large GrLivArea that are of a low price. 
#These values are huge oultliers. Therefore, we can safely delete them.
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# ### Note :
# Outliers removal is not always safe. We decided to delete these two as they are very huge and really bad ( extremely large areas for very low prices).
# There are probably others outliers in the training data. However, removing all them may affect badly our models if ever there were also outliers in the test data. That's why , instead of removing them all, we will just manage to make some of our models robust on them. You can refer to the modelling part of this notebook for that.

# ### Features engineering

# In[ ]:


#let's first concatenate the train and test data in the same dataframe

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
df_combined = pd.concat((train, test)).reset_index(drop=True)
df_combined.drop(['SalePrice'], axis=1, inplace=True)
print("df_combined size is : {}".format(df_combined.shape))


# In[ ]:


#Handling Missing Data

df_combined_na = (df_combined.isnull().sum() / len(df_combined)) * 100
df_combined_na = df_combined_na.drop(df_combined_na[df_combined_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :df_combined_na})
missing_data.head(30)


# In[ ]:


# plotting the missing data 

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=df_combined_na.index, y=df_combined_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[ ]:


#Correlation map to see how features are correlated with SalePrice

corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()


# In[ ]:


#Imputing missing values
# Using for loop to fill each category based data description

for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish',
           'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
           'MasVnrType','MSSubClass'):
        df_combined[col].fillna('NA', inplace=True) 


# In[ ]:


for col in ('GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
             'TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea'):
        df_combined[col].fillna(0, inplace=True) 


# In[ ]:


for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Functional','Utilities'):
        df_combined[col].fillna(df_combined[col].mode()[0], inplace=True) 


# In[ ]:


# Using .groupby and lambda expression to fill 'LotFrontage' column using 'Neighborhood 
df_combined['LotFrontage'] = df_combined.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


#Check remaining missing values if any 
df_combined_na = (df_combined.isnull().sum() / len(df_combined)) * 100
df_combined_na = df_combined_na.drop(df_combined_na[df_combined_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :df_combined_na})
missing_data.head(30)


# In[ ]:


#Transforming some numerical variables that are really categorical
#MSSubClass = The building class
df_combined['MSSubClass'] = df_combined['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
df_combined['OverallCond'] = df_combined['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
df_combined['YrSold'] = df_combined['YrSold'].astype(str)
df_combined['MoSold'] = df_combined['MoSold'].astype(str)


# In[ ]:


#Label Encoding some categorical variables that may contain information in their ordering set
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df_combined[c].values)) 
    df_combined[c] = lbl.transform(list(df_combined[c].values))

# shape        
print('Shape df_combined: {}'.format(df_combined.shape))


# In[ ]:


# Adding total sqfootage feature 
df_combined['TotalSF'] = df_combined['TotalBsmtSF'] + df_combined['1stFlrSF'] + df_combined['2ndFlrSF']


# In[ ]:


#
numeric_feats = df_combined.dtypes[df_combined.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df_combined[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[ ]:


# Drawing  
f, (ax1,ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10,figsize=(7,36))
sns.distplot(df_combined['MiscVal'], fit=norm, ax=ax1)
sns.distplot(df_combined['PoolArea'], fit=norm, ax=ax2)
sns.distplot(df_combined['LotArea'], fit=norm, ax=ax3)
sns.distplot(df_combined['LowQualFinSF'], fit=norm, ax=ax4)
sns.distplot(df_combined['3SsnPorch'], fit=norm, ax=ax5)
sns.distplot(df_combined['LandSlope'], fit=norm, ax=ax6)
sns.distplot(df_combined['KitchenAbvGr'], fit=norm, ax=ax7)
sns.distplot(df_combined['BsmtFinSF2'], fit=norm, ax=ax8)
sns.distplot(df_combined['EnclosedPorch'], fit=norm, ax=ax9)
sns.distplot(df_combined['ScreenPorch'], fit=norm, ax=ax10)


# In[ ]:


#Box Cox Transformation of (highly) skewed features
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df_combined[feat] += 1
    df_combined[feat] = boxcox1p(df_combined[feat], lam)
    
df_combined[skewed_features] = np.log1p(df_combined[skewed_features])


# In[ ]:


#Getting dummy categorical features 
df_combined = pd.get_dummies(df_combined, drop_first=True)
print(df_combined.shape)


# In[ ]:


#Getting the new train and test sets.
df_train = df_combined[:ntrain]
df_test = df_combined[ntrain:]


# ## Modelling 

# In[ ]:


#Import librairies
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import time
from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


X_train = df_train.values


# In[ ]:


X_test = df_test.values


# In[ ]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# ## Base Models

# In[ ]:


#LASSO Regression 
#This model may be very sensitive to outliers. So we need to made it more robust on them. 
#For that we use the sklearn's Robustscaler() method on pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


# In[ ]:


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


#ENet Regression
# Again made robust to outliers
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# In[ ]:


score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score


# In[ ]:


#define a rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


lasso.fit(X_train, y_train)
lasso_train_pred = lasso.predict(X_train)
lasso_pred = np.expm1(lasso.predict(X_test))
print(rmsle(y_train, lasso_train_pred))


# In[ ]:


#Test Submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = lasso_pred
sub.to_csv('submissionL.csv',index=False)


# ## Fine_Tuned Models

# In[ ]:


#Random Forest Regression using RandomizedSearchCV
rfr = RandomForestRegressor(criterion='mse', bootstrap=True, random_state=0, n_jobs=2)

param_dist = dict(n_estimators=list(range(1,100)),
                  max_depth=list(range(1,100)),
                  min_samples_leaf=list(range(1,10)))

rand = RandomizedSearchCV(rfr, param_dist, cv=10, verbose=1, n_iter=30)
rand.fit(X_train,y_train)


# In[ ]:


rand.best_score_ # RandomizedSearchCV


# In[ ]:


best_rfr = rand.best_estimator_


# In[ ]:


best_rfr.score(X_train,y_train) # RandomForestRegressor


# In[ ]:


rfr_train_pred = best_rfr.predict(X_train)
rfr_pred = np.expm1(best_rfr.predict(X_test))
print(rmsle(y_train, rfr_train_pred))


# In[ ]:


#Test Submission
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = rfr_pred
sub.to_csv('submissionRR.csv',index=False)


# ## Boston house price dictionary
# |Feature|Score|
# |---|---|
# |**LASSO Regression**|*0.1151*|
# |**Elastic Net**|0.1150|
# |**RandomForestRegressor**|*0.9666*|
# |**RandomizedSearchCV**|*0.8779567*|
# 

# In[ ]:




