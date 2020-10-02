#!/usr/bin/env python
# coding: utf-8

# Hello everyone! In this kernel my main goal is to work on outliers and see how they can affect our models, of course, there are many ways to deal with outliers such as z score, IQR and so on, also in some cases we can distinguish outliers manually, but at all it is not a good way to do that, I will try to review a few methods in this kernel in the future but for now I,m going to use pyod library which is one of a kind to distinguish outliers, this library uses algorithms like KNN and some other algorithms to take care of this task, well let's give it a try and see what happen. 
# First I will try to fix missing values problem in house price dataset then I will pay to the main issue, hope this will be useful. 
# 

# In[ ]:


pip install pyod


# In[ ]:


pip install impyute


# In[ ]:


import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# read the trainset 

# In[ ]:


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


train.head()


# Here I use two help functions, one is for return features with hight percentage null values and one just gives a description about datatype, number of null value and percentage of null values

# In[ ]:


def return_drop_list(df,percentage):
    temp = df.isnull().sum().sort_values(ascending=False)/len(df)*100
    drop_list = list(temp[temp >= percentage].index)
    return drop_list

def return_data_describe(df):
    
    types = df.dtypes
    null = df.isnull().sum().sort_values(ascending = False)
    percent = df.isnull().sum().sort_values(ascending = False)/len(df)*100
    
    data_describe = pd.DataFrame(types,columns = ['dtypes'])
    data_describe.insert(data_describe.shape[1],'num_of_null',null)
    data_describe.insert(data_describe.shape[1],'percent_of_null',percent)
    
    return data_describe.sort_values(by='num_of_null',ascending=False)


# In[ ]:


data_describe =  return_data_describe(train)
data_describe[data_describe.num_of_null >= 1]


# In[ ]:


temp = data_describe[data_describe.num_of_null>=1]
sns.set_style('darkgrid')
plt.figure(figsize=(12,8))
sns.barplot(y= temp.index,x = temp.percent_of_null )


# I,m going to drop features with more than 60 percent of null values 

# In[ ]:


drop_list = return_drop_list(train,60)
drop_list


# In[ ]:


train.drop(drop_list,axis=1,inplace=True)


# In[ ]:


data_describe =  return_data_describe(train)
data_describe[data_describe.num_of_null >= 1]


# According to data description we have to fill missing values with No for columns listed below,
# 
# **'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond','MasVnrType','FireplaceQu'** 

# In[ ]:


No_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond','MasVnrType','FireplaceQu']
for col in No_cols:
    train[col].fillna('No',inplace=True)


# In[ ]:


data_describe =  return_data_describe(train)
data_describe[data_describe.num_of_null >= 1]


# i use fast_knn algorithm from impyute library to fill missing values for columns **'LotFrontage','GarageYrBlt','MasVnrArea'**

# In[ ]:


cols = ['LotFrontage','GarageYrBlt','MasVnrArea']

from impyute.imputation.cs import fast_knn  

train[cols] = fast_knn(train[cols], k=20) 


# In[ ]:


data_describe =  return_data_describe(train)
data_describe[data_describe.num_of_null >= 1]


# In[ ]:


train.Electrical.fillna(method='ffill',inplace=True)


# columns **'MSSubClass','OverallCond','YrSold','MoSold'** are categorical with numeric values, 

# In[ ]:


cat_num_cols = ['MSSubClass','OverallCond','YrSold','MoSold']
for col in cat_num_cols:
    train[col] = train[col].astype(str)


# Lets distinguish catgegorical columns with numeric values from categorical columns with string values , I,m doing this because I will use label Encoding for categorical columns with string values.  

# In[ ]:


cat_str_cols = []
cat_numeric_col = []
for col in train.select_dtypes(include='object').columns:
    if str(train[col][0]).isnumeric() == False:
        cat_str_cols.append(col)
    else:
        cat_numeric_col.append(col)
        
label = LabelEncoder()
for col in cat_str_cols:
    train[col] = label.fit_transform(train[col])
    train[col] = train[col].astype(str)    


# To use KNN algorithm from pyod library I need to know which columns have high correlation with target (SalePrice), then lets see which one of them have hight correlation with target

# In[ ]:


train.corr()['SalePrice'].abs().sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(10,6),dpi=200)
sns.heatmap(train.corr())


# In[ ]:


corr = train.corr()['SalePrice'].abs().sort_values(ascending=False)
h_corr_cols = corr[corr >= .5].index.tolist()


# **boxplot** is of the ways that we can see outliers clearly

# In[ ]:


import random
from matplotlib.colors import cnames
colors = list(cnames.keys())
sns.set_style('darkgrid')
fig , ax = plt.subplots(3,4,figsize = (16,12))
ax = ax.ravel()
for i,col in enumerate(h_corr_cols):
    sns.boxplot(train[col], ax = ax[i],color = random.choice(colors))


# I use columns with high correlation with the target to train KNN algorithm and then use this algorithms to predict which rows in our dataset are outliers, 

# In[ ]:


x = train[h_corr_cols].values
model = KNN(contamination=.1)
model.fit(x)
predicted = model.predict(x)

outliers = train.loc[(predicted == 1),:]
inliers = train.loc[(predicted == 0),:]


# To see outliers I,m going to use **scatterplot**,  

# In[ ]:


sns.set_style('darkgrid')

fig, ax = plt.subplots(3,4,figsize=(20,10),dpi=150)
ax= ax.ravel()
for i,col in enumerate(h_corr_cols):
    sns.scatterplot(x='SalePrice',y=col,data = train.loc[(predicted == 1),:],ax=ax[i],color='r')
    sns.scatterplot(x='SalePrice',y=col,data = train.loc[(predicted == 0),:],ax=ax[i],color='b')


# 
# To see more clearly I choose one of features, the left plot shows data with outliers and the right side shows data after cleaning the outliers  
# 

# In[ ]:


sns.set_style('darkgrid')
fig , ax  = plt.subplots(1,2,figsize=(12,6),dpi=150)
ax = ax.ravel()

ax[0].scatter(inliers.GrLivArea,inliers.SalePrice,edgecolors='k',c='b',label='inliers')
ax[0].scatter(outliers.GrLivArea,outliers.SalePrice,edgecolors='k',c='r',label='outliers')
ax[1].scatter(inliers.GrLivArea,inliers.SalePrice,edgecolors='k',c='b',label='inliers')


ax[0].set_xlabel('GrLivArea')
ax[0].set_ylabel('SalePrice')
ax[1].set_xlabel('GrLivArea')
ax[1].set_ylabel('SalePrice')

ax[0].legend(loc= 'upper left')
ax[1].legend(loc= 'upper left')


# also 3d plot can show ouliers very well 

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
sns.set_style('whitegrid')

fig = plt.figure(figsize=(16,10),dpi=300)
ax = fig.add_subplot(111, projection='3d')


x_out = outliers.GrLivArea
y_out = outliers.OverallQual
z_out = outliers.SalePrice

x_in = inliers.GrLivArea
y_in = inliers.OverallQual
z_in = inliers.SalePrice

ax.scatter(x_out, y_out, z_out, c='r', marker='o',edgecolor='k',label='outliers')
ax.scatter(x_in, y_in, z_in, c='b', marker='o',edgecolor='k',label='inliers')

ax.set_xlabel('GrLivArea')
ax.set_ylabel('verallQual')
ax.set_zlabel('SalePrice')
ax.legend()


# In[ ]:


train_w_outlier = train.drop(index = train.loc[(predicted == 1),:].index ) # drop outliers from trainset


# boxplot for SalePrice with outliers and without outliers

# In[ ]:


sns.set_style('darkgrid')
fig , ax = plt.subplots(1,2,figsize = (9,4))
ax = ax.ravel()

sns.boxplot(train['SalePrice'], ax = ax[0],color=random.choice(colors))
sns.boxplot(train_w_outlier['SalePrice'], ax = ax[1],color=random.choice(colors))


# Well, we have a dataset with ouliers and one with outliers , let's see how outliers can affect our data to be skewed,here i use help function to draw skew plots   

# In[ ]:


from scipy.stats import  skew
num_features = train.select_dtypes(exclude='object').columns
def skew_plot(df):
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(5,7,figsize=(16,10))
    ax = ax.ravel()
    for i,col in enumerate(num_features):
        sk = df[col].skew()
        if sk >= .8:
            color ='r'
        else:
            color = 'g'
        ax[i] = sns.distplot(df[col], ax = ax[i],color=color)
        ax[i].set_xlabel(col)
          
        
def apply_log_transform(df):
    num_features = df.select_dtypes(exclude='object').columns
    for col in num_features:
        if df[col].skew() >= .8:
            df[col] = np.log1p(df[col])
    return df


# skew plot with outliers 

# In[ ]:


skew_plot(train)


# skew plots without outliers

# In[ ]:


skew_plot(train_w_outlier)


# I want to train models and see scores (errors) with outliers and without outliers 

# In[ ]:


train.drop('Id',axis=1,inplace=True)
train_w_outlier.drop('Id',axis=1,inplace=True)

y = train['SalePrice']
y_w_outliers = train_w_outlier['SalePrice']

x = train.drop('SalePrice',axis=1)
x_w_outliers = train_w_outlier.drop('SalePrice',axis=1)


# notice that here i use [Serigne](http://https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) notebook parametrs for xgb.XGBRegressor

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)



model_RF = RandomForestRegressor(n_estimators=100)


# In[ ]:


def preprocess(X): 
    temp = pd.get_dummies(X,drop_first=True)
    standard = StandardScaler()
    finall = standard.fit_transform(temp)
    return finall

def score2(model,x,y):
    
    error =  cross_val_score(model,
                               preprocess(x),y,
                               scoring='neg_mean_absolute_error').mean()
    return np.abs(error).round(2)


# In[ ]:


score2(model_xgb,apply_log_transform(x),y) # error with outliers


# In[ ]:


score2(model_xgb,apply_log_transform(x_w_outliers),y_w_outliers) # error without outliers


# In[ ]:


score2(model_RF,apply_log_transform(x),y) # error with outliers


# In[ ]:


score2(model_RF,apply_log_transform(x_w_outliers),y_w_outliers) #  error without outliers


# In[ ]:




