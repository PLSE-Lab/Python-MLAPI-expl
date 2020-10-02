#!/usr/bin/env python
# coding: utf-8

# # Objective: 
# Predict the house price based on 79 variables. We are going to use the advance regression technique to achieve the output.
# 
#  - Step-1 Importing the Data
#  - Step-2 Inspecting the dataframe
#  - Step-3 Data Quality Checks and Cleaning
#  - Step-4 Exploratory Analysis on Data
#  - Step-5 Model Building and Evaluation
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# ### Step-1 Importing the data

# In[3]:


# reading the dataset
df = pd.read_csv("../input/train.csv")
tf = pd.read_csv("../input/test.csv")
df.head()


# ### Step-2 Inspecting the Data Frame

# In[ ]:


# Shape and Size of Data

print(df.shape)
print(df.info())


# In[ ]:


# Shape and Size of Data

print(tf.shape)
print(tf.info())


# In[ ]:


df1=df.iloc[:,1:40]


# In[ ]:


# Check Missing Values
round(100*(df1.isnull().sum()/len(df1.index)), 2)


# In[ ]:


df2=df.iloc[:,41:81]
# Check Missing Values
round(100*(df2.isnull().sum()/len(df2.index)), 2)


# ### Step-3 - Data Quality Checks and Cleaning

# In[ ]:


# LotFrontage - Replace NULL with Mean value
df['LotFrontage'].value_counts()
df['LotFrontage'].mean()


# In[ ]:


tf['LotFrontage'].value_counts()
tf['LotFrontage'].mean()


# In[ ]:


df['LotFrontage'].replace(np.nan,70.0,inplace= True)
df['LotFrontage'].value_counts().head()


# In[ ]:


tf['LotFrontage'].replace(np.nan,68.5,inplace= True)
tf['LotFrontage'].value_counts().head()


# In[ ]:


# Alley  - Drop this var, it has 90% values as missing
df['Alley'].value_counts()


# In[ ]:


# Alley  - Drop this var, it has 90% values as missing
tf['Alley'].value_counts()


# In[ ]:


# MasVnrType - Replace with None
df['MasVnrType'].value_counts()
df['MasVnrType'].replace(np.nan,"None",inplace= True)


# In[ ]:


# MasVnrType - Replace with None
tf['MasVnrType'].value_counts()
tf['MasVnrType'].replace(np.nan,"None",inplace= True)


# In[ ]:


# MasVnrArea - Replace with 0.0
df['MasVnrArea'].value_counts()
df['MasVnrArea'].replace(np.nan,0.0,inplace= True)


# In[ ]:


# MasVnrArea - Replace with 0.0
tf['MasVnrArea'].value_counts()
tf['MasVnrArea'].replace(np.nan,0.0,inplace= True)


# In[ ]:


# FireplaceQu - Drop this variable due to very high missing values
df['FireplaceQu'].value_counts()


# In[ ]:


# FireplaceQu - Drop this variable due to very high missing values
tf['FireplaceQu'].value_counts()


# In[ ]:


# GarageType - fill with Attchd
df['GarageType'].value_counts()


# In[ ]:


# GarageType - fill with Attchd
tf['GarageType'].value_counts()


# In[ ]:


df['GarageType'].replace(np.nan,"Attchd",inplace= True)
df['GarageType'].value_counts()


# In[ ]:


tf['GarageType'].replace(np.nan,"Attchd",inplace= True)
tf['GarageType'].value_counts()


# In[ ]:


# GarageYrBlt - Not so important drop this vars
df['GarageYrBlt'].value_counts().head()


# In[ ]:


# GarageYrBlt - Not so important drop this vars
tf['GarageYrBlt'].value_counts().head()


# In[ ]:


# GarageFinish 
df['GarageFinish'].value_counts()


# In[ ]:


# GarageFinish 
tf['GarageFinish'].value_counts()


# In[ ]:


df['GarageFinish'].replace(np.nan,"Unf",inplace= True)
df['GarageFinish'].value_counts()


# In[ ]:


tf['GarageFinish'].replace(np.nan,"Unf",inplace= True)
tf['GarageFinish'].value_counts()


# In[ ]:


# GarageQual - Replace with TA
df['GarageQual'].value_counts()


# In[ ]:


# GarageQual - Replace with TA
tf['GarageQual'].value_counts()


# In[ ]:


df['GarageQual'].replace(np.nan,"TA",inplace= True)
df['GarageQual'].value_counts()


# In[ ]:


tf['GarageQual'].replace(np.nan,"TA",inplace= True)
tf['GarageQual'].value_counts()


# In[ ]:


# GarageCond 
df['GarageCond'].value_counts()


# In[ ]:


# GarageCond 
tf['GarageCond'].value_counts()


# In[ ]:


df['GarageCond'].replace(np.nan,"TA",inplace= True)
df['GarageCond'].value_counts()


# In[ ]:


tf['GarageCond'].replace(np.nan,"TA",inplace= True)
tf['GarageCond'].value_counts()


# In[ ]:


# PoolQC  - Delete this var
df['PoolQC'].value_counts()


# In[ ]:


# PoolQC  - Delete this var
tf['PoolQC'].value_counts()


# In[ ]:


# Fence - Delete this var
df['Fence'].value_counts()


# In[ ]:


# Fence - Delete this var
tf['Fence'].value_counts()


# In[ ]:


# MiscFeature - Delete this var
df['MiscFeature'].value_counts()


# In[ ]:


# MiscFeature - Delete this var
tf['MiscFeature'].value_counts()


# In[ ]:


# BsmtQual - Replace with TA
df['BsmtQual'].value_counts()


# In[ ]:


# BsmtQual - Replace with TA
tf['BsmtQual'].value_counts()


# In[ ]:


df['BsmtQual'].replace(np.nan,"TA",inplace= True)


# In[ ]:


tf['BsmtQual'].replace(np.nan,"TA",inplace= True)


# In[ ]:


# BsmtCond - Replace with TA
df['BsmtCond'].value_counts()


# In[ ]:


# BsmtCond - Replace with TA
tf['BsmtCond'].value_counts()


# In[ ]:


df['BsmtCond'].replace(np.nan,"TA",inplace= True)
df['BsmtCond'].value_counts()


# In[ ]:


# BsmtExposure - Replace with No
df['BsmtExposure'].value_counts()


# In[ ]:


# BsmtExposure - Replace with No
tf['BsmtExposure'].value_counts()


# In[ ]:


df['BsmtExposure'].replace(np.nan,"No",inplace= True)
df['BsmtExposure'].value_counts()


# In[ ]:


tf['BsmtExposure'].replace(np.nan,"No",inplace= True)
tf['BsmtExposure'].value_counts()


# In[ ]:


# BsmtFinType1 - Replace with Unf
df['BsmtFinType1'].value_counts()


# In[ ]:


# BsmtFinType1 - Replace with Unf
tf['BsmtFinType1'].value_counts()


# In[ ]:


df['BsmtFinType1'].replace(np.nan,"Unf",inplace= True)
df['BsmtFinType1'].value_counts()


# In[ ]:


tf['BsmtFinType1'].replace(np.nan,"Unf",inplace= True)
tf['BsmtFinType1'].value_counts()


# In[ ]:


# check for Electrical
df['Electrical'].value_counts()
df['Electrical'].replace(np.nan,"SBrkr",inplace= True)


# In[ ]:


# check for Electrical
tf['Electrical'].value_counts()
tf['Electrical'].replace(np.nan,"SBrkr",inplace= True)


# In[ ]:


df['Electrical'].value_counts()


# In[ ]:


tf['Electrical'].value_counts()


# In[ ]:


# drop not required columns
df_1=df.drop(columns=['Alley','FireplaceQu','GarageYrBlt','PoolQC','Fence','MiscFeature'])
df_1.head()


# In[ ]:


# drop not required columns
tf_1=tf.drop(columns=['Alley','FireplaceQu','GarageYrBlt','PoolQC','Fence','MiscFeature'])
tf_1.head()


# In[ ]:


#Check missing values again 
round(100*(df_1.isnull().sum()/len(df_1.index)), 2)


# In[ ]:


df_1.shape


# In[ ]:


#Check missing values again 
pd.set_option('display.max_columns', 100)
round(100*(tf_1.isnull().sum()/len(tf_1.index)), 2)


# In[ ]:


# chek for duplicate values for unique id
tf_1['Id'].nunique()
# No duplicate values here


# In[ ]:


# MSZoning - Replace NULL with Mean value
tf_1['MSZoning'].value_counts()
tf_1['MSZoning'].replace(np.nan,"RL",inplace= True)


# In[ ]:


# Utilities - Replace NULL with Mean value

tf_1['Utilities'].replace(np.nan,"AllPub",inplace= True)
tf_1['Utilities'].value_counts()


# In[ ]:


tf_1['Exterior1st'].value_counts()
tf_1['Exterior1st'].replace(np.nan,"VinylSd",inplace= True)


# In[ ]:


# Check Missing Values
tf_1 = tf_1.dropna(how='any')
tf_1.shape


# In[ ]:


df_1 = df_1.dropna(how='any')


# In[ ]:


df_1.shape


# ### Step-4 - Exploratory Analysis on Data

# In[ ]:


sns.distplot(df_1['SalePrice'])


# In[ ]:


print("Skewness: %f" % df_1['SalePrice'].skew())
print("Kurtosis: %f" % df_1['SalePrice'].kurt())


# In[ ]:


var = 'GrLivArea'
data = pd.concat([df_1['SalePrice'], df_1[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


var = 'TotalBsmtSF'
data = pd.concat([df_1['SalePrice'], df_1[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_1['SalePrice'], df_1[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


var = 'YearBuilt'
data = pd.concat([df['SalePrice'], df_1[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# In[ ]:


#correlation matrix
corrmat = df_1.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_1[cols], size = 2.5)
plt.show();


# ### Step-4- Data Preparation and Test, Train Split

# In[ ]:


sns.distplot(df_1["SalePrice"])


# In[ ]:


sns.distplot(np.log(df["SalePrice"]))


# - Sales is skwed towards right convert it in logarthimic scale

# In[ ]:


df_1["TransformedPrice"] = np.log(df_1["SalePrice"])


# In[ ]:


types_train = df_1.dtypes #type of each feature in data: int, float, object
num_train = types_train[(types_train == 'int64') | (types_train == float)] #numerical values are either type int or float
cat_train = types_train[types_train == object] #categorical values are type object


# In[ ]:


types_train = tf_1.dtypes #type of each feature in data: int, float, object
num_train1 = types_train[(types_train == 'int64') | (types_train == float)] #numerical values are either type int or float
cat_train1 = types_train[types_train == object] #categorical values are type object


# In[ ]:


pd.DataFrame(types_train).reset_index().set_index(0).reset_index()[0].value_counts()


# In[ ]:


numerical_values_train = list(num_train.index)


# In[ ]:


print(numerical_values_train)


# In[ ]:


numerical_values_train = list(num_train1.index)
print(numerical_values_train)


# In[ ]:


categorical_values_train = list(cat_train.index)
print(categorical_values_train)


# In[ ]:


categorical_values_train = list(cat_train1.index)
print(categorical_values_train)


# In[ ]:


# split into X and y
X = df_1.iloc[:,1:74]

y = df_1['TransformedPrice']


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X.shape


# In[ ]:


y = df_1['TransformedPrice']
y.head()


# In[ ]:


y.shape


# In[ ]:


# subset all categorical variables
df_cat_1 = df_1.select_dtypes(include=['object'])
df_cat_1.head()


# In[ ]:


# subset all categorical variables
tf_cat_1 = tf_1.select_dtypes(include=['object'])
tf_cat_1.head()


# In[ ]:


# convert into dummies
df_cat_dummies = pd.get_dummies(df_cat_1, drop_first=True)
df_cat_dummies.head()


# In[ ]:


# convert into dummies
tf_cat_dummies = pd.get_dummies(tf_cat_1, drop_first=True)
tf_cat_dummies.head()


# In[ ]:


# drop categorical variables 
X = X.drop(list(df_cat_1.columns), axis=1)


# In[ ]:


# concat dummy variables with X
X = pd.concat([X, df_cat_dummies], axis=1)
X.head()


# In[ ]:


# drop categorical variables 
tf_1 = tf_1.drop(list(tf_cat_1.columns), axis=1)
# concat dummy variables with X
tf_2 = pd.concat([tf_1, df_cat_dummies], axis=1)


# In[ ]:


tf_2.head()


# In[ ]:


tf_2 = tf_2.dropna(how='any')
tf_2.shape


# In[ ]:


tf_id=tf_2.iloc[:,0]

tf_id=pd.DataFrame(tf_id)
tf_id.head()


# In[ ]:



tf_3=tf_2.iloc[:,1:228]
tf_3.head()


# In[5]:


# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# ### Step-5- Model Building and Evaluation

# ### Lasso Regression

# In[ ]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
lasso = Lasso()
folds = 5
# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[ ]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[ ]:


model_cv.best_params_


# In[ ]:


alpha = .001

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 


# In[ ]:


lasso.coef_


# ### Ridge Regression

# In[ ]:


# Applying Ridge
ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()


# In[ ]:


# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()


# In[ ]:


model_cv.best_params_


# In[ ]:


alpha = 20
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_


# ### Step-6- Submission

# In[ ]:


#Using Lasso regression because it support feature selection and provids more gernelize model


# In[ ]:


pred=lasso.predict(tf_3)
preds=np.exp(pred)


# In[ ]:


output=pd.DataFrame({'Id':tf_id.Id, 'SalePrice':preds})


# In[ ]:


output.head()


# In[ ]:


output.to_csv('submission.csv', index=False)


# In[ ]:


output.shape

