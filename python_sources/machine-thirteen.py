#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Introduction
# 
# The aim of this notebook is to predict the sale price of houses in Ames Iowa given features of a house. In order to do this, we will need statistical libraries, the dataset which we sourced from www.kaggle.com/c/house-prices-advanced-regression-techniquesand ,perform exploratory data analysis and feature engineering on the dataset to prepare our data for the model. We have chosen to use 3 techniques namely Ridge, Lasso and ENet to train and predict our dataset. We will then decide which one best predicts our test dataset.

# # Importing additional libraries and the dataset

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Exploratory Data Analysis
# The Id column serves no purpose in our model so we drop it.

# In[3]:


train.drop('Id', inplace = True, axis =1)
test.drop('Id', inplace = True, axis =1)


# #Let's look at what our train and test data looks like.

# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


#variable change for later use
test_A = test


# # Correlation using Heapmap
# We want to visualise pairwise correlations between independent variables and the dependent variable

# In[7]:


corr_overall = train.corr()
k=12

col = corr_overall.nlargest(k, 'SalePrice')['SalePrice'].index
corr_coeff = np.corrcoef(train[col].values.T)
ax = plt.subplots(figsize=(20,15))
heatmap = sns.heatmap(corr_coeff, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':12}, yticklabels=col.values, xticklabels=col.values)


# We see that Overall Quality of the house and GrLivArea (Ground living area) are highly correlated with the saleprice by 0.79 and 0.71 respectively. We use these two independent variables to inspect any unusual observations.

# # Removing outliers

# We inspect the scatter plot for any outliers in OverallQual vs SalePrice. But first let's look at the shape of our dataset

# In[8]:


train.shape


# In[9]:


fig, plot = plt.subplots()
plot.scatter(x= train['OverallQual'], y= train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('OverallQual', fontsize=13)
plt.show()


# We decide to not remove any outliers for this because they do not look to be too extreme and the price of the house could be because of the ground living area as this also influences house prices.
# 
# GrLivArea = 1stFlrSF + 2ndFlrSF and this is a good indicator for predicting a house price. So we try and look for outliers to remove.

# In[10]:


fig, plot = plt.subplots()
plot.scatter(x= train['GrLivArea'], y= train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We observe extreme outliers in this relationship where the ground living area is between 4000 and 6000 and the sale price is under 300000. We need to remove these kinds of unusual observation in our dataset because they can cause overfitting.

# In[11]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index) #drop rows that meet this criteria


# In[12]:


fig, plot = plt.subplots()
plot.scatter(x= train['GrLivArea'], y= train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We check our train size again and we find that 2 rows have been removed.

# In[13]:


train.shape


# # Preparing the data for cleaning
# We first merge the train and test dataset and drop the Id column because it has no value to our model.

# In[14]:


train_A = train.drop(['SalePrice'], axis=1)
y = pd.DataFrame(train['SalePrice'])
features = pd.concat([train_A, test_A]).reset_index(drop=True)


# Now we look at our new merged dataset

# In[15]:


features.head()


# In[16]:


#function to plot the distribution of a variable. Returns the distribution, skeweness and kurtosis
def distribution(df,column_name):
    
    sns.distplot(df[column_name], color = 'b',kde = True)
    plt.title('Distribution of ' + column_name)
    plt.xlabel(column_name)
    plt.ylabel('Number of occurences')
    
    #skewness
    skewness = df[column_name].skew()
    if (skewness > -0.5) & (skewness < 0.5):
        print('The data is fairly symmetrical with skewness of ' + str(skewness))
    elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):
        print('The data is moderately skewed with skewness of ' + str(skewness))
    elif (skewness < -1) | (skewness > 1):
        print('The data is highly skewed with skewness of ' + str(skewness))
    #kurtosis    
    print('The kurtosis is ' + str(df[column_name].kurt()))


# Kurtose is the measure of peakness in the distribution. Identifying outliers calls for investigating why the data contains such extreme values.It Could be incorrect entries or other things to help us understand the data better or even remove this incorrect information. Note that we have already removed 2 outliers in our training dataset, we now look at the distribution of the SalePrice.

# In[17]:


distribution(y,'SalePrice')


# It looks like we still have extreme values. The distribution is skewed to the right. In statistics it it better to work with known distributions when we are trying to make predictions. Our dataset is enough for us to normalise it. We use the log function to try and normalise our distribution.

# In[18]:


y_log = pd.DataFrame(np.log1p(y['SalePrice']))
distribution(y_log,'SalePrice')


# Our response variable is now somewhat normal which is good for our model accuracy because we will now work with a known distribution.

# We now separate our data into numerical and categorical data.

# In[19]:


numerical_features = features.dtypes[features.dtypes != "object"].index
categorical_features = features.dtypes[features.dtypes == "object"].index

numerical_df = features[numerical_features]
categorical_df = features[categorical_features]


# # Visualising missing data
# 
# We plot the missing data to show how many missing values we will have to deal with.

# In[20]:


features_na = (features.isnull().sum() / len(features)) * 100
#drop features without missing values
features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :features_na})

#plot
f, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x=features_na.index, y=features_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent', fontsize=15)
plt.xticks(rotation='90')
plt.title('Percent of missing data by feature', fontsize=15)


# Looking at the barplot we see that some features look to have 0 missing features. This is not the case. Those are features that have 1 or just a few missing values.

# # Functions to replace missing values
# 
# Some Nan values are missing and some are actually 0 because the feature is not present.

# In[21]:


#mean
def fill_na_num_mean(df,column_name):

    df[column_name] = df[column_name].transform(lambda x: x.fillna(round(x.mean(),1)))
    
    return df
#median
def fill_na_num_median(df,column_name):

    df[column_name] = df[column_name].transform(lambda x: x.fillna(round(x.median(),1)))
    
    return df

#Nan to None
def fill_na_cat_none(df,column_name):
    df[column_name].fillna('None',inplace=True)
    
#Nan to 0 
def fill_na_num_0(df,column_name):
    df[column_name].fillna(0,inplace=True)
    
#mode
def fill_na_mode(df,column_name):
    df[column_name].fillna(df[column_name].mode()[0], inplace = True)
    


# # Distributions of independent variables
# 
# Before we replace missing values we would first like to visualise the distributions of the independent variables.

# In[22]:


quantitative = [f for f in features.columns if features.dtypes[f] != 'object'] #numerical
qualitative = [f for f in features.columns if features.dtypes[f] == 'object'] #categorical

f = pd.melt(features, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")


# # Replace missing values

# From the above plots, We see that MiscFeature will not serve our model any purpose so we remove it as well as MiscValue (most of these are 0)

# In[23]:


features.drop(['MiscFeature'], inplace = True, axis = 1)
features.drop(['MiscVal'], inplace = True, axis = 1)


# There are too many Nans that mean None for Fence, Alley and PoolQc. We decide to remove these variables because they will not affect our saleprice much.

# In[24]:


features.drop(['Fence'], inplace = True, axis =1)
features.drop(['Alley'], inplace = True, axis = 1)
features.drop(['PoolQC'], inplace = True, axis = 1)
features.drop(['PoolArea'], inplace = True, axis =1)


# Lets now look at LotFrontage. We investigate the distribution so we can make an informed decision on how to replace the missing values
# 

# In[25]:


plt.hist(features['LotFrontage'])
plt.show()

#skewness
skewness = features['LotFrontage'].skew()
if (skewness > -0.5) & (skewness < 0.5):
    print('The data is fairly symmetrical with skewness of ' + str(skewness))
elif ((skewness > -1) & (skewness < -0.5))| ((skewness > 0.5) & (skewness < 1)):
    print('The data is moderately skewed with skewness of ' + str(skewness))
elif (skewness < -1) | (skewness > 1):
    print('The data is highly skewed with skewness of ' + str(skewness))
        
#kurtosis
print('The kurtosis is ' + str(features['LotFrontage'].kurt()))


# The distribution is highly skewed and there are outliers. Replacing with the mean doesnt seem to be the best strategy since this is not a normal distribution. We can therefore think about this in terms of the area of interest. Usually housing developments around the same area have the same lot area and frontage. We can use the mode to replace these missing values.

# In[26]:


fill_na_mode(features,'LotFrontage')


# Let's now dive into replacing missing values. We focus more on numerical data first, but we will also check relating categorical columns simultaneously.
# 
# We replace Nan with mode for MasVnrType, replace Nan with 0 for MasVnrArea

# In[27]:


features[['MasVnrArea','MasVnrType']].head()


# In[28]:


fill_na_cat_none(features,'MasVnrType') 
fill_na_num_0(features,'MasVnrArea')


# Here NaN mean NONE-no basement. sq feet here is 0 for all NaN BsmtFinSF2,correlating column is also NaN for presence of bsmt
# Nan replaced with None

# In[29]:


features[['BsmtFinSF2','BsmtFinType2']].head()


# Nan also replaced with 0 for sq feet

# In[30]:



fill_na_num_0(features,'BsmtFinSF2')

features[['BsmtFinSF1','BsmtFinType1']].head()


#  If there is no basement present the value of the square feet is zero

# In[31]:



features[['BsmtUnfSF','BsmtFinType1','BsmtFinType2']].head()


# from the above we see that BsmtFinType1 and BsmtFinType2 has None as a category from the data description provided.
# For the square feet we replace with 0 for corresponding basement type.

# In[32]:


fill_na_cat_none(features,'BsmtFinType1')
fill_na_num_0(features,'BsmtFinSF1')
fill_na_cat_none(features,'BsmtFinType2')
fill_na_num_0(features,'BsmtFinSF2')

features[['TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']].head()


# Total sq feet for multiple basements including unfinished. Total is dependent on 3 variables
# Nan in Total due to no basement is replaced with 0.

# In[33]:


fill_na_num_0(features,'TotalBsmtSF')
fill_na_num_0(features,'BsmtUnfSF')


# We now check whether there is a basement by checking the value of the basement square feet in order to replace Nan values or baths in the basement.

# In[34]:


features[['BsmtFullBath','BsmtHalfBath','TotalBsmtSF']].head()


# In[35]:


fill_na_num_0(features,'BsmtFullBath')
fill_na_num_0(features,'BsmtHalfBath')


# NaN means no garage in categorical. so 0 for corresponding columns
# One detatched garageType but NaN everywhere. decision to keep value as 0.
# 
# We would also like to drop GarageYrBlt because it has less influence on the house price.
# We also drop Utilities because almost all records are "AllPub" except for one which is 'NoSeWa' and 2 NAs, this is in the training set, therefore will not affect the predictions of the test dataset.

# In[36]:


features[['GarageType','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].head()


# In[37]:


features.drop('GarageYrBlt', inplace = True, axis = 1)
features.drop(['Utilities'], axis=1, inplace = True)

fill_na_cat_none(features,'GarageType')
fill_na_cat_none(features,'GarageFinish')
fill_na_cat_none(features,'GarageQual')
fill_na_cat_none(features,'GarageCond')


fill_na_num_0(features,'GarageCars')
fill_na_num_0(features,'GarageArea')


# We are now done with numerical data. We Fill the remaining missing data

# In[38]:


new_numerical_features = features.dtypes[features.dtypes != "object"].index

new_categorical_features = features.dtypes[features.dtypes == "object"].index

new_num_df = features[new_numerical_features]
new_cat_df = features[new_categorical_features]

features[['BsmtQual','BsmtCond','BsmtExposure','TotalBsmtSF']].head()


# Basement exposure has NaN on line 948,1498,2348,1487, but theres a basement.missing value
# line 2524,2185,2040 bsmscond NaN but there is a basement
# NaN on BsmntQl but theres a basement, line 2217,2218
# 
# Solution: replace all valaues with most frequent where Total area of basement is not 0
# cols most likely correlated, could predict outcome of the other?
# 
# saletype-most frequent
# 

# In[39]:


fill_na_cat_none(features,'BsmtExposure')
fill_na_cat_none(features,'BsmtCond')
fill_na_cat_none(features,'BsmtQual')
fill_na_mode(features,'MSZoning')
fill_na_mode(features,'SaleType')
fill_na_mode(features,'KitchenQual')
fill_na_mode(features,'Electrical')
fill_na_mode(features,'Exterior1st')
fill_na_mode(features,'Exterior2nd')
fill_na_mode(features,'Functional')

features[['FireplaceQu','Fireplaces']].head()


# In[40]:


fill_na_cat_none(features,'FireplaceQu')


# # Creating dummy variables

# In[41]:


features.head()


# In[42]:


#encounding categorical data (ordinal)
features['LandContour'] = features['LandContour'].replace(dict(Lvl=4, Bnk=3, HLS=2, Low=1))
features['LandSlope'] = features['LandSlope'].replace(dict(Gtl=3, Mod=2, Sev=1))
features['ExterQual'] =features['ExterQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['ExterCond'] =features['ExterCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['BsmtQual'] = features['BsmtQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['BsmtCond'] =features['BsmtCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['BsmtCond'] =features['BsmtCond'].replace('None',0)
features['BsmtExposure'] =features['BsmtExposure'].replace(dict(Gd=4, Av=3, Mn=2, No=1))
features['BsmtExposure'] =features['BsmtExposure'].replace('None',0)
features['BsmtFinType1'] = features['BsmtFinType1'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1))
features['BsmtFinType1'] =features['BsmtFinType1'].replace('None',0)
features['BsmtFinType2'] = features['BsmtFinType2'].replace(dict(GLQ=6, ALQ=5, BLQ=4, Rec=3, LwQ=2, Unf=1))
features['BsmtFinType2'] = features['BsmtFinType2'].replace('None',0)
features['HeatingQC'] = features['HeatingQC'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['CentralAir'] = features['CentralAir'].replace(dict(Y=1, N=0))
features['KitchenQual'] =features['KitchenQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['Functional'] = features['Functional'].replace(dict(Typ=8, Min1=7, Min2=6, Mod=5, Maj1=4, Maj2=3, Sev=2, Sal=1))
features['FireplaceQu'] = features['FireplaceQu'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['FireplaceQu'] = features['FireplaceQu'].replace('None', 0)
features['GarageQual'] = features['GarageQual'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['GarageQual'] =features['GarageQual'].replace('None', 0)
features['GarageCond'] = features['GarageCond'].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1))
features['GarageCond'] = features['GarageCond'].replace('None', 0)
features['LotShape'] = features['LotShape'].replace(dict(Reg=4, IR1=3, IR2=2, IR3=1))
features['GarageFinish'] = features['GarageFinish'].replace(dict(Fin=3, RFn=2, Unf=1))
features['GarageFinish'] = features['GarageFinish'].replace('None', 0)
features['PavedDrive'] =features['PavedDrive'].replace(dict(Y=3, P=2, N=1))
features =features.astype({"Functional": int})


# In[43]:


features['BsmtQual'] =features['BsmtQual'].replace('None',0)#assigning 0 to none to turn the object into an int
features =features.astype({"KitchenQual": int})
features =features.astype({"Functional": int})


# In[44]:


#dummy variables of none ranked columns
features['MSSubClass'] =features['MSSubClass'].astype('category')
features['MoSold'] = features['MoSold'].astype('category')
features['YrSold'] =features['YrSold'].astype('category')
cat_cols = ['MSZoning','Street','MSSubClass', 'MoSold', 'YrSold', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']
dumies=pd.get_dummies(features[cat_cols],drop_first=True)
dumies.head()


# In[45]:


for column in cat_cols:
    features.drop([column],axis=1,inplace=True)


# In[46]:


features=pd.concat([ features,dumies],axis=1)
features.head()


# Our variables are no longer objects

# In[47]:


features.dtypes


# # Preparing the dataset for the model
# We need to separate our dataset back into train and test
# 
# Looking at the data set we see that the scale is different. We need to standardise our variables to have the same scale to better predict the response variable.

# In[48]:


new_train = features.iloc[:1458, :]
new_test = features.iloc[1458:,:]

X_test = new_test
X_train = new_train

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)


# Lets now look at our new dataset for the training data.

# In[49]:


X_standardize = pd.DataFrame(X_scaled, columns = X_train.columns)
X_standardize.head()


# The dataset looks to be on a similar scale. We also see that the all have the same standard deviation, which is good for our prediction.

# In[50]:


X_standardize.describe().loc['std']


# # Model Selection
# 
# For this dataset, we will use 3 techniques to predict the test response variable and choose the one which best predicts this dependent variable.

# # Ridge regression

# In[51]:


from sklearn.linear_model import Ridge
from sklearn import metrics

ridge = Ridge()
ridge.fit(X_train, y_log)

y_sale = ridge.predict(X_test)
y_predict_ridge = np.expm1(y_sale)
submission = pd.DataFrame(y_predict_ridge)

#creating submission file
sample = pd.read_csv('../input/sample_submission.csv')
sample['SalePrice']=submission[0]
sample.to_csv('Ridge.csv',index = False)


# In[52]:


#train predict
y_train = ridge.predict(X_train)


# # Lasso

# In[53]:


from sklearn.linear_model import Lasso,ElasticNet
from sklearn import metrics


# In[54]:


lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_log)

y_lasso_pred = lasso.predict(X_test)
lasso_predict = np.expm1(y_lasso_pred)
submission = pd.DataFrame(lasso_predict)

#creating submission file
sample = pd.read_csv('../input/sample_submission.csv')
sample['SalePrice']=submission[0]
sample.to_csv('Lasso.csv',index = False)


# In[55]:


y_lasso_train = lasso.predict(X_train)


# 

# # Enet

# In[56]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

ENet.fit(X_train, y_log)
ENet_train_pred = ENet.predict(X_train)

ENet_pred = np.expm1(ENet.predict(X_test))
sub = pd.DataFrame(ENet_pred)

sub.head()

#creating submission file
sample = pd.read_csv('../input/sample_submission.csv')
sample['SalePrice']=sub[0]
sample.to_csv('Enet.csv',index = False)


# In[57]:


ridge.score(X_train,y_log)


# In[58]:


ENet.score(X_train,y_log)


# In[59]:


lasso.score(X_train,y_log)


# # MSE
# 

# In[60]:


print('Ridge MSE: ',metrics.mean_squared_error(y_log,y_train))


# In[61]:


print('Enet MSE: ',metrics.mean_squared_error(y_log,ENet_train_pred))


# In[62]:


print('Lasso MSE: ',metrics.mean_squared_error(y_log,y_lasso_train))


# In[63]:


plt.scatter(np.arange(len(y_log)), ENet_train_pred, label='Predicted')
plt.scatter(np.arange(len(y_log)), y_log, label='Training')

plt.legend()
plt.show()


# # Conclusion
# 
# We see that the R^2 value of Ridge is better than the rest as well as the Mean square error of the train. But Ridge did not perform better than Enet on the test response variable on kaggle. So we have chosen Enet as our Technique that best predicted the SalePrice. The scatter plot shows that our predictions are not far off from the actual sale price in the train dataset.

# In[ ]:




