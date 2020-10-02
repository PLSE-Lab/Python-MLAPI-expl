#!/usr/bin/env python
# coding: utf-8

# ## Vuyo Mpondo ##
# 
# 

# # House Price Prediciton Model #

# The Kaggle dataset below provides us with information on how to predict a house price based on a number of features. This data set was analyzed and modeled using regression methods in order to develop a procedure that used to accurately predict the selling price of a house. The model we used is a weighted combination of the ElasticNet and Ridge regression models. This provided our model with accuracy and stability in order to be consistent accross all future datasets.

# Fistly we importing all the relevant libraries,including the regression models that we will be using. Numpy, Pandas and SciPy are the libraries used for mathematical, statistical and dataframe manipulation. Pyplot and Seaborn allow us to generate the necessary plot 
# 
# As stated below we will be using Elasticnet and Ridge regressors. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from scipy import stats
from scipy.stats import skew
from sklearn.linear_model import ElasticNet, Ridge

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


display(HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
    }
"""))


# We imported the data into a Train and Test dataframe. 

# In[ ]:


#reading the train and test data
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# Analysing the SalePrice data, we noticed that the data was Right-Skewed. This would make it harder to develop a reliable prediction model. We apply log transformation to normalize the distribution.

# In[ ]:





# In[ ]:


#plotting the salesprice distribution
ax1 = sns.distplot(train['SalePrice'])
ax1.set_title('Distplot Normal Distribution')

#Creating QQ - plot
fig = plt.figure()
sx1 = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#applying np.log to distribution
train['SalePrice'] =np.log1p(train['SalePrice'])

#New salesprice distribution(After np.log)
ax2 = sns.distplot(train['SalePrice'])
ax2.set_title('Distplot after log transformation')

#New QQ plot Distribution(After np.log)
fig = plt.figure()
sx2 = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# We created a seperate variable to save the SalePrice data to train the model

# In[ ]:


#Creating y-train data
y_train = train.SalePrice.values


# For the data exploration and imputation, we decided to combine the train and test data. This would allow for consistency on data handling and further manipulation

# In[ ]:


#combining train and test data
df = pd.concat([train,test], sort=False)


# We split the columns into categorical and numerical features based on the tyoe of data in the column.

# In[ ]:


#Getting datatypes for each category
df.dtypes

#classifying numerical and categorical features
numFeats = df.dtypes[df.dtypes != "object"].index
catFeats = df.dtypes[df.dtypes == "object"].index
print(numFeats)
print(catFeats)


# The numerical variables showed that some of the numerical variables also showed a skewness to some degree. We applied the log transformation to the columns that showed a skewness of 75% or more.

# In[ ]:


#log transform skewed numeric features:
skewed_feats = train[numFeats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

df[skewed_feats] = np.log1p(df[skewed_feats])


# ATTENDING TO MISSING VALUES
# 
# We analysed the data in the column to determine the amount of missing data in both sets. 

# In[ ]:


#Counts percentage of null values for each column
for x in df:
    if round(df[x].isna().mean() * 100, 2) > 0:
        print(x,  round(df[x].isna().mean() * 100, 2),'%' )


# Together with the data_description document provided in the data set, we analysed each feature for the most likely variable to enter into the NaN items. Missing data did not necessarily indicate that the information was not provided, in some cases, such as the PoolQC, it can indicate that that particular house does not have that feature, which will have to be filled with a new category, 'None'. Other features were more suited to be filled by the most common value in the case of categorical features and the mean in the case of the numerical variables. 

# In[ ]:


#Numerical value imputation
for x in df[numFeats]:
    df[x].fillna((round(df[x].mean(),0)), inplace=True)
    
#MANUAL CATEGORICAL IMPUTATION
#Most Frequent Value Imputation
df[ 'BsmtCond' ].fillna('TA' , inplace = True)
df[ 'BsmtExposure' ].fillna( 'No', inplace = True)
df[ 'Electrical' ].fillna('SBrkr' , inplace = True)
df[ 'Exterior1st' ].fillna('VinylSd' , inplace = True)
df[ 'Exterior2nd' ].fillna('VinylSd' , inplace = True)
df[ 'ExterCond' ].fillna('TA' , inplace = True)
df[ 'ExterQual' ].fillna('TA' , inplace = True)
df[ 'Functional' ].fillna('Typ' , inplace = True)
df[ 'KitchenQual' ].fillna('TA' , inplace = True)
df['MSZoning'].fillna('RL', inplace = True)
df[ 'SaleType' ].fillna( 'WD', inplace = True)


#####New Category 'None' Imputation
df[ 'Alley' ].fillna('None' , inplace = True)
df[ 'BsmtFinType1' ].fillna( 'None', inplace = True)
df[ 'BsmtFinType2' ].fillna( 'None', inplace = True)
df[ 'BsmtQual' ].fillna('None' , inplace = True)
df[ 'Fence' ].fillna('None' , inplace = True)
df[ 'FireplaceQu' ].fillna('None' , inplace = True)
df[ 'Foundation' ].fillna('None' , inplace = True)
df[ 'GarageCond' ].fillna( 'None', inplace = True)
df[ 'GarageFinish' ].fillna( 'None', inplace = True)
df[ 'GarageQual' ].fillna( 'None', inplace = True)
df[ 'GarageType' ].fillna('None' , inplace = True)
df[ 'MasVnrType' ].fillna('None' , inplace = True)
df[ 'MiscFeature' ].fillna( 'None', inplace = True)
df[ 'PoolQC' ].fillna('None' , inplace = True)


# The correlation graphs provide us with the features that are most likely to affect the final sale price, as well as features that will affect each other. Based on the Correlation graphs, the Box Plot for Categorical Variables and the RegPlots for the Numerical Variables, we selected columns we felt were important and influential to our final regression model.

# In[ ]:


# Heatmap for correlations between variables
corrmat = df.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

# Heatmap for most correlated variables
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.35]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot=True)


# In[ ]:


# Box plot for categorical variables
li_cat_feats = ['Alley','LotShape','Exterior1st','Exterior2nd','BsmtFinType1', 'BsmtFinType2', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2',
      'HouseStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'Heating',
      'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','FireplaceQu', 'GarageType',
      'GarageFinish', 'GarageQual','PavedDrive','SaleType', 'SaleCondition']
target = 'SalePrice'
nr_rows = 7
nr_cols = 4
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))
for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y=target, data=df, ax = axs[r][c])
plt.tight_layout()    
plt.show()


# In[ ]:


# Scatter Plot for numerical variables
li_num_feats = ['3SsnPorch','LotFrontage', 'BedroomAbvGr','LotArea', 'OverallQual','YearBuilt', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2',
        'TotalBsmtSF','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','GrLivArea', 'FullBath','Fireplaces', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'OverallCond','TotRmsAbvGrd','WoodDeckSF', 'PoolArea']   
target = 'SalePrice'
nr_rows = 6
nr_cols = 4
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))
for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_num_feats):
            sns.regplot(x=li_num_feats[i], y=target, data=df,  ax = axs[r][c])
    
plt.tight_layout()    
plt.show()


# A new dataframe is created with the selected columns

# In[ ]:


X = df[li_num_feats+li_cat_feats]


# Dummy variables are generated to seperate the categorical variables. 

# In[ ]:


X = pd.get_dummies(X, drop_first =True)


# The dataframe is seperated into train and test data.

# In[ ]:


#Split train and test data
X_train = X.iloc[:1460, :]
X_test = X.iloc[1460:, :]


# We selected the Robust Scaler to scale the features.

# In[ ]:


sc=RobustScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# Create the Ridge model Regressor and fit it to the training data.

# In[ ]:


ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_pred = np.expm1(ridge_pred)


# Create the ElasticNet model Regressor and fit it to the training data.

# In[ ]:


ENet =ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
ENet.fit(X_train, y_train)
ENet_pred = ENet.predict(X_test)
ENet_pred = np.expm1(ENet_pred)


# Select the weighted averages to provide the Optimized final model. 

# In[ ]:


final_model = ((ridge_pred)*0.30 + (ENet_pred)*0.7)

