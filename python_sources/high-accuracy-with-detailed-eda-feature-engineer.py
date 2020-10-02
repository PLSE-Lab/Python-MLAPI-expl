#!/usr/bin/env python
# coding: utf-8

# # Detailed EDA + Feature Engineering + Feature Selection + Model Building
# 
# **If you find this notebook useful or even liked than you can greatly upvate to motivate me further....**

# 
# 
# I have always believed that as long as you do good feature engineering on your data you dont need fancy algorithms to improve your model performance. This is exactly what i have done in this notebook. I have used simple algirithms yet have got quite high accuracy score.
# 
# 
# 
# ## Any data science projecct involves following steps
# 1. Data Analysis
# 1. Feature Engineering
# 1. Feature Selection
# 1. Model Building
# 1. Model Deployment
# 
# We have followed exactly these steps excluding last step;.
# 
# ### 1. Data Analysis
# we have performed given analyis in in this section. 
# 1. Missing Values
# 1. All The Numerical Variables
# 1. Distribution of the Numerical Variables
# 1. Categorical Variables
# 1. Cardinality of Categorical Variables
# 1. Outliers
# 1. Relationship between independent and dependent feature(SalePrice)
# 
# ### 2. Feature Engineering
# We have done following in this section:
# 
# 1. Missing values handling
# 2. Outlier handling
# 3. Normalization
# 4. Label Encoding/One Hot Encoding
# 4. Feature Scaling etc..
# 
# ### 3. Feature Selection
# Feature selction means selecting those features that have improved performance on our model. I have used some Machine Learning and Statistical methods to select most promising features for improved model performance.
# 
# ### 4. Model Building
# This is the last section in our notebook .i have used simple algorithms   Linear  Regression, RandomForestRegressor etc to build model. You can't believe with good feature engineering i was able to build pretty good model.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# lets load data
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


# create copy of above dataframe. 
train = train_df.copy()
test = test_df.copy()


# In[ ]:


# lets see shape of datas
print('train data shape: ', train.shape)
print('test data shape: ', test.shape)


# In[ ]:


# lets view first five records in train data
train.head()


# In[ ]:


# lets view first 5 observations in test data
test.head()


# In[ ]:


# you can see null values even in firstt 5 observations as seen above
# lets find the null values in data

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


# Id attributes have no special meaning in in regression so lets drop them
train.drop('Id', axis=1, inplace = True)
test.drop('Id', axis=1, inplace=True)


# ## 1. Detailed EDA

# **Let us first check the distribution of target feature**

# In[ ]:


## plotting distribution of target feature
sns.distplot(train['SalePrice'])
plt.show()


# Target feature is not normally distributed and shows positive skewness. We need to do some transformation like log normal which se wll do in feature engineering section

# ### 1.1 Numerical Features

# In[ ]:


# Numerical features
Numerical_feat = [feature for feature in train.columns if train[feature].dtypes != 'O']
print('Total numerical features: ', len(Numerical_feat))
print('\nNumerical Features: ', Numerical_feat)


# In[ ]:


# making a glance of first 5 observations
train[Numerical_feat].head()


# In[ ]:


# Zoomed heatmap, correlation matrix
sns.set(rc={'figure.figsize':(12,8)})
correlation_matrix = train.corr()

k = 10             #number of variables for heatmap
cols = correlation_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# From above heatmap we can select those dependent features which have high correlations with target feature but low correlation among dependent features. these selected features given below in next cell.

# In[ ]:


## these are selected features from heatmap
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']


# **1.1.1 Discrete Features**

# In[ ]:


# Discrete features

discrete_feat = [feature for feature in Numerical_feat if len(train[feature].unique())<25]
print('Total discrete features: ', len(discrete_feat))
print('\n', discrete_feat)


# In[ ]:


# glancing first five records of discrete features
train[discrete_feat].head()


# In[ ]:


train[discrete_feat].info()


# In[ ]:


# Lets find unique values in each discrete features
for feature in discrete_feat:
    print('Uique values of ', feature, ':')
    print(train[feature].unique())
    print('\n')
    


# In[ ]:


## Lets Find the realtionship between discrete features and SalePrice

#plt.figure(figsize=(8,6))

for feature in discrete_feat:
    data=train.copy()
    plt.figure(figsize=(8,6))
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# **1.1.2 Continuous Features**

# In[ ]:


continuous_features = [feature for feature in Numerical_feat if feature not in discrete_feat]
print('The numbers of continuous features: ', len(continuous_features))
print('\n', continuous_features)


# In[ ]:


## Lets analyse the continuous values by creating histograms to understand the distribution

train[continuous_features].hist(bins=25)
plt.show()


# They shows quite a skewness.

# In[ ]:


## let us now examine the relationship between continuous features and SalePrice
## Before that lets find continous features that donot contain zero values

continuous_nozero = [feature for feature in continuous_features if 0 not in data[feature].unique() and feature not in ['YearBuilt', 'YearRemodAdd']]

for feature in continuous_nozero:
    plt.figure(figsize=(8,6))
    data = train.copy()
    data[feature] = np.log(data[feature])
    data['SalePrice'] = np.log(data['SalePrice'])
    plt.scatter(data[feature], data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.show()


# In[ ]:


## Normality and distribution checking for continous features
for feature in continuous_nozero:
    plt.figure(figsize=(6,6))
    data = train.copy()
    sns.distplot(data[feature])
    plt.show()


# Above plots shows skewness, peakness and no normality. We need to dosome transformations like log noraml which we will do in Feature Engineering section.

# ### 1.2 Categorical Features

# In[ ]:


# categorical features
categorical_feat = [feature for feature in train.columns if train[feature].dtypes=='O']
print('Total categorical features: ', len(categorical_feat))
print('\n',categorical_feat)


# In[ ]:


# lets view few samples 
train[categorical_feat].head()


# In[ ]:


# lets find unique values in each categorical features
for feature in categorical_feat:
    print('{} has {} categories. They are:'.format(feature,len(train[feature].unique())))
    print(train[feature].unique())
    print('\n')


# In[ ]:


# let us find relationship of categorical with target variable

for feature in categorical_feat:
    data=train_df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:





# ## 2. Feature Engineering

# Alright, we will now combine train set and test set. since we are doing feature engineering here, combining them together and performing feature engineering will save time from having to repeat the same process for test data.

# In[ ]:


Train = train_df.shape[0]
Test = test_df.shape[0]
target_feature = train_df.SalePrice.values
combined_data = pd.concat((train_df, test_df)).reset_index(drop=True)
combined_data.drop(['SalePrice','Id'], axis=1, inplace=True)
print("all_data size is : {}".format(combined_data.shape))


# ### 2.1 Missing data and handling them.

# In[ ]:


# let's find the missing data in combined dataset

total = combined_data.isna().sum().sort_values(ascending=False)
percent = (combined_data.isnull().sum()/combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# ### 2.1.1 Numerical features(handling missing data)
# We shall deal with missing datas in numerical features here.

# In[ ]:


# Lets first handle numerical features will nan value
numerical_nan = [feature for feature in combined_data.columns if combined_data[feature].isna().sum()>1 and combined_data[feature].dtypes!='O']
numerical_nan


# In[ ]:


combined_data[numerical_nan].isna().sum()


# In[ ]:


## Replacing the numerical Missing Values

for feature in numerical_nan:
    ## We will replace by using median since there are outliers
    median_value=combined_data[feature].median()
    
    combined_data[feature].fillna(median_value,inplace=True)
    
combined_data[numerical_nan].isnull().sum()


# ### 2.1.2 Categorical features(handling missing data)
# WE shall handle missing datas in this section for categorical features.

# In[ ]:


# categorical features with missing values
categorical_nan = [feature for feature in combined_data.columns if combined_data[feature].isna().sum()>1 and combined_data[feature].dtypes=='O']
print(categorical_nan)


# In[ ]:


combined_data[categorical_nan].isna().sum()


# In[ ]:


# replacing missing values in categorical features
for feature in categorical_nan:
    combined_data[feature] = combined_data[feature].fillna('None')


# In[ ]:


combined_data[categorical_nan].isna().sum()


# In[ ]:





# ### 2.2 Outliers
# Outliers are the data points that just deviates from other normal data points. Outliers can have a great effect in performance of ML models. So we have to be very careful handling them. Removing outliers always may not be good choice, we should see the nature of data and other aspects when handling them.

# In[ ]:


# these are selected features from EDA section
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']


# In[ ]:


# plot bivariate distribution (above given features with saleprice(target feature))
for feature in features:
    if feature!='SalePrice':
        plt.scatter(train_df[feature], train_df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# We can see a clear otliers in **GrLivAreea** and **TotalBsmtSF**. I mean it just doesn't make sense for larger values **GrLivAreea** and **TotalBsmtSF** to have low value of SalePrice. There might be some reason for this but we wll consider them outlier here and drop them.

# In[ ]:


#Deleting outliers for GrLivArea
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

plt.scatter(train_df['GrLivArea'], train_df['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()


# In[ ]:


#Deleting outliers for TotalBsmtSF
train_df = train_df.drop(train_df[(train_df['TotalBsmtSF']>5000) & (train_df['SalePrice']<300000)].index)

plt.scatter(train_df['TotalBsmtSF'],train_df['SalePrice'])
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.show()


# ### 2.3 Normalizing some numerical data
# We know some numerical data shows skewness, we will normalize/transform them to normal distribution by using log normal transformation

# In[ ]:


# these are selected features from EDA section
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# selecting continuous features from above
continuous_features = ['SalePrice', 'GrLivArea', 'TotalBsmtSF']


# In[ ]:


# checking distribution of continuous features(histogram plot)
for feature in continuous_features:
    if feature!='SalePrice':
        sns.distplot(combined_data[feature], fit=norm)
        plt.show()
    else:
        sns.distplot(train_df['SalePrice'], fit=norm)
        plt.show()


# In[ ]:


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
# This idea is from Pedro Marcelino, PhD notebook.

combined_data['HasBsmt'] = 0  # at first o for all observations in 'HasBsmt'
combined_data.loc[combined_data['TotalBsmtSF']>0,'HasBsmt'] = 1  # assign 1 for those with no basement 


# In[ ]:


#transform data
combined_data.loc[combined_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(combined_data['TotalBsmtSF'])
combined_data['GrLivArea'] = np.log(combined_data['GrLivArea'])
train_df['SalePrice'] = np.log(train_df['SalePrice'])


# In[ ]:


# we have log transormed above skewed data. Now lets see their distribution
for feature in continuous_features:
    if feature!='SalePrice':
        sns.distplot(combined_data[feature], fit=norm)
        plt.show()
    else:
        sns.distplot(train_df['SalePrice'],fit=norm)
        plt.show()


# We can see we have attained a bit of normality here. There are also other few continuous features that might have skewness but above features have more effect on target features so we have only considered them here.

# In[ ]:





# ### 2.4 Label encoding, One-Hot-Encoding/dummies
# We will label encode some features and perform one hot encoding on categorical features that are not ordinal(doesnot show any information in order form).
# But yes, there are some features given as numerical/discrete numerical but actually looks like categorical which gives information in order form for example 'Overallcond' feature rates the overall condition of the house in range 1 to 10.

# In[ ]:


## these are features that seems to give information in order form
## taken from Serigne's notebook

ordinal_features = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
                 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
                 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
                 'YrSold', 'MoSold']
print(len(ordinal_features))


# In[ ]:


## Credit for Serigne 

#MSSubClass=The building class
combined_data['MSSubClass'] = combined_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
combined_data['OverallCond'] = combined_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
#combined_data['YrSold'] = combined_data['YrSold'].astype(str)
#combined_data['MoSold'] = combined_data['MoSold'].astype(str)


# In[ ]:


combined_data[ordinal_features].head(10)


# If we look at the above table we can see some of the features are numeric/discrete but in realty they can be consdered categorical, its jsut that they are avalable in encoded form in this dataset.
# so when we label encode above ordinal_features, those features that are numeric/discrete remans unchanged as they are already in encoded form. The 

# In[ ]:


# so let's label encode above ordinal features
from sklearn.preprocessing import LabelEncoder
for feature in ordinal_features:
    encoder = LabelEncoder()
    combined_data[feature] = encoder.fit_transform(list(combined_data[feature].values))


# In[ ]:


# Now lets see label encoded data
combined_data[ordinal_features].head()


# In[ ]:


## One hot encoding or getting dummies 

dummy_ordinals = pd.get_dummies(ordinal_features) 
dummy_ordinals.head()


# In[ ]:


# creating dummy variables

combined_data = pd.get_dummies(combined_data)
print(combined_data.shape)


# In[ ]:


combined_data.head(10)


# ### 2.5 Feature Scaling
# Many machine learning algorithms especially in Regression, models seems to have pooer performance on unscaled data. So what we will do here is scale the data in same range between 0 and 1. Even if we scale the distance between points still remains unchanged.

# In[ ]:


# let's first see descriptive stat info 
combined_data.describe()


# we can see above data range differs so much. so we need to scale them to same range.

# In[ ]:


## we willtake all features from combined_dummy_data 
features_to_scale = [feature for feature in combined_data]
print(len(features_to_scale))


# In[ ]:


## Now here is where we will scale our data using sklearn module.

from sklearn.preprocessing import MinMaxScaler

cols = combined_data.columns  # columns of combined_dummy_data

scaler = MinMaxScaler()
combined_data = scaler.fit_transform(combined_data[features_to_scale])


# In[ ]:


# after scaling combined_data it is now in ndarray datypes
# so we will create DataFrame from it
combined_scaled_data = pd.DataFrame(combined_data, columns=[cols])


# In[ ]:


combined_scaled_data.head() # this is the same combined_dummy_data in scaled form.


# In[ ]:


# lets see descriptive stat info 
combined_scaled_data.describe()


# We can see from above two dataframe tables that datas are now scaled.

# In[ ]:


train_df.shape, test_df.shape, combined_scaled_data.shape, combined_data.shape


# Initially, train data had **1460** observations but we had droped 2 oo 3 in outlier handling section so now we have **4581** observations.

# In[ ]:


# separate train data and test data 
train_data = combined_scaled_data.iloc[:1460,:]
test_data = combined_scaled_data.iloc[1460:,:]

train_data.shape, test_data.shape


# In[ ]:


## lets add target feature to train_data
train_data['SalePrice']= train_df['SalePrice']  # This saleprice is normalized. Its very impportant


# In[ ]:


train_data = train_data
train_data.head(10)


# In[ ]:


test_data = test_data.reset_index()
test_data.tail(10)


# In[ ]:





# In[ ]:


## ugh.. it seems outliers that we droped earlier haven't droped from combined data.
## that makes sense since we had droped only from train data before not from combined data.
## S0 we will drop them here

#Deleting outliers for TotalBsmtSF
#train_data = train_data.drop(train_data[(train_data['TotalBsmtSF']>5000) & (train_data['SalePrice']<300000)].index)

#Deleting outliers for GrLivArea
#train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)


# In[ ]:





# ### 3 Feature Selection and Model Building
# Here we will first select those feature that are most omportant for our models and we shall build models using them

# ### 3.1 Feature selection

# In[ ]:


dataset = train_data.copy()  # copy train_data to dataset variable


# In[ ]:


dataset.head()


# In[ ]:


dataset = dataset.dropna()


# In[ ]:


## lets create dependent and target feature vectors

X = dataset.drop(['SalePrice'],axis=1)
Y = dataset[['SalePrice']]

X.shape, Y.shape


# In[ ]:


Y.head()


# In[ ]:


# lets do feature selection here

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# define feature selection
fs = SelectKBest(score_func=f_regression, k=27)
# apply feature selection
X_selected = fs.fit_transform(X, Y)
print(X_selected.shape)


# We can see that 30 best/important features have been selected. 

# In[ ]:


cols = list(range(1,28))

## create dataframe of selected features

selected_feat = pd.DataFrame(data=X_selected,columns=[cols])
selected_feat.head()


# I haven't included which features are selected. just know that 30 features are selected

# In[ ]:





# In[ ]:


# perform train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(selected_feat,Y,test_size=0.3,random_state=0)


# In[ ]:


x_train.shape, x_test.shape


# ### 3.2 Model Building
# We will build machine learning models using above selected features

# ### 3.2.1 Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics

lr = LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


y_pred = lr.predict(x_test) # predicting test data
y_pred[:10]


# In[ ]:


# Evaluating the model
print('R squared score',metrics.r2_score(y_test,y_pred))

print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#  R square score is preety high almost 90% score which is preety good.
#  MAE, MSE and RMSE values also shows pretty good result.

# In[ ]:


# check for underfitting and overfitting
print('Train Score: ', lr.score(x_train,y_train))
print('Test Score: ', lr.score(x_test,y_test))


# Above train score and test score  comparable which is good. Even though it shows a slight case of underfitting but thats fine here.

# In[ ]:


## scatter plot of original and predicted target test data
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.xlabel('y_tes')
plt.ylabel('y_pred')
plt.show()


# In[ ]:


## Lets do error plot
## to get error in prediction just substract predicted values from original values

error = list(y_test.values-y_pred)
plt.figure(figsize=(8,6))
sns.distplot(error)


# ### 3.2.2 RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(x_train,y_train)


# In[ ]:


y_pred = rf_reg.predict(x_test)
print(y_pred[:10])


# In[ ]:


## evaluating the model

print('R squared error',metrics.r2_score(y_test,y_pred))

print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


# check score
print('Train Score: ', rf_reg.score(x_train,y_train))
print('Test Score: ', rf_reg.score(x_test,y_test))


#  We have train set accuracy of **0.9792356047371404** and test set accuracy of **0.8848913083086402**. Here we can see overfiiting issue but for now we will leave it alone. They are still preety good score.

# In[ ]:


## scatter plot of original and predicted target test data
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.xlabel('y_tes')
plt.ylabel('y_pred')
plt.show()


# In[ ]:


## Lets do error plot
## to get error in prediction just substract predicted values from original values

error = list(y_test.values-y_pred)
plt.figure(figsize=(8,6))
sns.distplot(error)


# Such a nice error plot. We can see the errors are normally distributed.

# **Valueable suggestions and Feedbacks are always welcomed, if you find this notebook useful or liked than you can upvote this work. Thank you everyone for visiting this notebook.**

# In[ ]:




