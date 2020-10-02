#!/usr/bin/env python
# coding: utf-8

# ## House Prices: Advanced Regression Techniques
# 
# The Notebook is based on a Kaggle competition with the following description. Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# ### EDA: Exploratory Data Analysis
# 
# As the description suggest we probably want to have a better undestanding of the data before diving into a complex Machine Learning algorithm. We start by reading our data into a Pandas dataframe and finding the type of the variables in it.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


sample=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train_copy=train.copy()
test_copy=test.copy()


# In[ ]:


numeric=train.select_dtypes(include=[np.number]) #selecting the data asociate with the numerical variables


# In[ ]:


categorical=train.select_dtypes(exclude=[np.number]) #selecting the data asociate with the categorical variables


# As we can see there are 38 numerical variables including SalePrice and Id:
# 
# ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
#        'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
#        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
#        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
#        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
#        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
#        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
#        'MiscVal', 'MoSold', 'YrSold', 'SalePrice']
#        
#  Similarly there are 43 categorical variables:
#  
#    ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
#        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
#        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
#        'SaleType', 'SaleCondition']
# 

# ### Exploring the Target variable; SalePrice
# 
# Normally most of the people can buy houses with low prices. Now let's see if this is reflected in the histogram of our target variable.

# In[ ]:


train['SalePrice'].hist()


# ### Skewness
# 
# It is the degree of distortion from the symmetrical bell curve or the normal distribution. It measures the lack of symmetry in data distribution.
# 
# If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
# 
# If the skewness is between -1 and -0.5(negatively skewed) or between 0.5 and 1(positively skewed), the data are moderately skewed.
# 
# If the skewness is less than -1(negatively skewed) or greater than 1(positively skewed), the data are highly skewed.
# 
# As it was expected our target variable is right skewed, therefore we have to normalize its distribution. There are many ways to approaching this but we have selected a simple one: logarithm transformation, but we are going to do this later.

# In[ ]:


y=train['SalePrice']


# We also eliminate the  target variable and the Id to form the feature set.

# In[ ]:


train.drop(['SalePrice','Id'], axis=1, inplace=True)


# In[ ]:


numeric.drop(['SalePrice','Id'], axis=1, inplace=True)
num_columns=numeric.columns


# We use the skew function to find the features variables wich are skewed and select between them the ones with high values based on the criteria presented before.

# In[ ]:


def select_skew_index(df):
    numeric=df.select_dtypes(include=[np.number])
    num_columns=numeric.columns
    skew_features = df[num_columns].skew(axis = 0, skipna = True)
    high_skewness = skew_features[skew_features > 0.5]
    skew_index = high_skewness.index
    return skew_index


# In[ ]:


skew_index=select_skew_index(train)


# In[ ]:


train[skew_index].hist(figsize=(15,15))


# #### Kurtosis
# It describe the extreme values in one versus the other tail and measures the outliers present in the distribution.
# 
# High kurtosis in a data set is an indicator that data has heavy tails or outliers. In this case we should investigate wrong data entry or other things.
# 

# In[ ]:


kurt_features = train[num_columns].kurtosis(axis = 0, skipna = True)


# In[ ]:


high_kurt = kurt_features[kurt_features > 3]
kurt_index = high_kurt.index
high_kurt


# #### Outliers
# 
# As we saw before it is a good approach to look for outlier in the variables with high values of kurtosis. To have a better idea let's draw the scatter plot of this variables.

# In[ ]:


fig=plt.figure(figsize=(15,20))
for i in range(1,18):
    ax=fig.add_subplot(6,3,i)
    ax.scatter(x=train[kurt_index[i-1]], y=y)
    ax.set_xlabel(kurt_index[i-1])


# From these graphics we can see that there is one value in 'BsmtFinSF1','GrLivArea' and 'TotalBsmtSF' each, with high values of square feets and low sale price. That has no sense, moreover, when find this value for 'BsmtFinSF1','GrLivArea' and 'TotalBsmtSF' it turns to be the same index 1298. Therefore, 1298 its our first candidate
# to outlier.

# In[ ]:


train[train['BsmtFinSF1']>5000].index


# In[ ]:


train[train['TotalBsmtSF']>6000].index


# In[ ]:


train[train['GrLivArea']>5000].index


# In[ ]:


train.shape


# In[ ]:


train=train.drop(train.index[1298])
y=y.drop(y.index[1298])


# In[ ]:


train.shape


# Now we want to correct the skewness and kurtosis by applying a logarithm transformation. We start with our target variable and then continue with the features

# In[ ]:


y=np.log1p(y)


# In[ ]:


y.hist()


# Now it looks much better. 

# In[ ]:


def correct_skew(df,skew_index):
    for i in skew_index:
        df[i] = np.log1p(df[i])


# In[ ]:


correct_skew(train,skew_index)


# In[ ]:


train[skew_index].hist(figsize=(15,15))


# #### Filling missing values
# We start by finding the variables with nan values

# In[ ]:


train.isnull().sum().sort_values(ascending=False)[0:25]


# First we have to look at the data description for a better understanding of the data. There are variables like Alley for which NA values means No alley access and variables like 'GarageCars' for which NA means 0.

# In[ ]:


def fill_miss(df):
    Nvalues=['FireplaceQu','GarageFinish','BsmtCond','Alley','BsmtExposure','GarageCond','PoolQC','BsmtQual',
             'MiscFeature','MasVnrType','BsmtFinType1','GarageType','Fence','GarageQual','BsmtFinType2']
    GarBsmt=['GarageYrBlt','GarageCars','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
'BsmtFullBath','BsmtHalfBath']
    df_cat=df.select_dtypes(exclude=[np.number])
    stats_df=df_cat.describe() 
    for i in df.columns:
        if(i in Nvalues):
            df[i].replace(np.nan,"None", inplace=True )
        elif(i in GarBsmt):
            df[i].replace(np.nan,0, inplace=True )
        
    for i in df_cat.columns:
        top = stats_df[i].iloc[2]
        if(df[i].isnull().sum()!=0):
            df[i].replace(np.nan,top, inplace=True )
            
    df.interpolate(inplace=True)


# In[ ]:


fill_miss(train)


# In[ ]:


train.isnull().sum()


# ### Categorical variables
# 
# Now we have to work with the categorical variables and select the ones that are ordinal and nominal to find a correct approach for them. For this we have to look the data description and find similarities between categories. We have notice that the variable MSSubClass is in fact categorical therefore we hace to change its type to object.

# Also we want to look for variables which have the majority of its elements in only one category, for this we have the following function:

# In[ ]:


def uniq_cat(df):
    categoric=df.select_dtypes(exclude=[np.number])
    cat_col=categoric.columns
    high_val=[]
    for i in cat_col:
        for j in range(df[i].unique().shape[0]):
            if ((df[i].value_counts()[j])/1459 > 0.99):
                 high_val.append(i) 
    return high_val


# In[ ]:


uniq_cat(train)


# In[ ]:


train = train.drop(['Utilities', 'Street', 'PoolQC',], axis=1)


# We change the variables that are ordinal according to the importance of the category. We combine the variables Condition1 and Condition2 into dummies to avoid duplicates later

# In[ ]:


def filling_ordinal(df):
    feat=['ExterQual','ExterCond','BsmtQual','BsmtCond','KitchenQual','HeatingQC','KitchenQual'
          ,'HeatingQC','GarageQual','GarageCond']
    for x in feat:  
        df[x][df[x] == 'Ex'] = 5
        df[x][df[x] == 'Gd'] = 4
        df[x][df[x] == 'TA'] = 3
        df[x][df[x] == 'Fa'] = 2
        df[x][df[x] == 'Po'] = 1
        df[x][df[x] == 'None'] = 0
        
    df['LandSlope'][df['LandSlope'] == 'Sev'] = 3
    df['LandSlope'][df['LandSlope'] == 'Mod'] = 2
    df['LandSlope'][df['LandSlope'] == 'Gtl'] = 1
    
    df['BsmtExposure'][df['BsmtExposure'] == 'Gd'] = 4
    df['BsmtExposure'][df['BsmtExposure'] == 'Av'] = 3
    df['BsmtExposure'][df['BsmtExposure'] == 'Mn'] = 2
    df['BsmtExposure'][df['BsmtExposure'] == 'No'] = 1
    df['BsmtExposure'][df['BsmtExposure'] == 'None'] = 0
    
    feat1=['BsmtFinType1','BsmtFinType2']
    
    for x in feat1:
        df[x][df[x] == 'GLQ'] = 6
        df[x][df[x] == 'ALQ'] = 5
        df[x][df[x] == 'BLQ'] = 4
        df[x][df[x] == 'Rec'] = 3
        df[x][df[x] == 'LwQ'] = 2
        df[x][df[x] == 'Unf'] = 1
        df[x][df[x] == 'None'] = 0
        
    df['CentralAir'][df['CentralAir'] == 'Y'] = 1
    df['CentralAir'][df['CentralAir'] == 'N'] = 0
    


# In[ ]:


filling_ordinal(train)


# ### Feature Ingeneering

# In[ ]:


def feat_ing(X):
    X['TotalBath']=X['BsmtFullBath']+ (1/2)*X['BsmtHalfBath']+X['FullBath']+ (1/2)*X['HalfBath']
    X['TotalSF']=X['TotalBsmtSF']+X['1stFlrSF']+X['2ndFlrSF']


# In[ ]:


feat_ing(train)


# In[ ]:


X=pd.get_dummies(train)


# In[ ]:


def clean(rtrain,rtest):
    y=rtrain['SalePrice']
    testId=rtest['Id']
    rtrain.drop(['Id','SalePrice'],axis=1,inplace=True)
    rtest.drop(['Id'],axis=1,inplace=True)
    
    # selecting the indexes of the skew features
    skew_index=select_skew_index(rtrain)
    
    # Eliminate the outier
    rtrain=rtrain.drop(rtrain.index[1298])
    y=y.drop(rtrain.index[1298])
    
    # Drop the columns in the test data with all values equal to na
    rtest=rtest.dropna(axis=1,how='all')
    
    # preparing features and target values
    y=np.log1p(y)

    #Correct the skewness
    #correct_skew(rtrain,skew_index)
    #correct_skew(rtest,skew_index)
    
    #Filling missing values
    fill_miss(rtrain)
    fill_miss(rtest)
    
    #Drop the features with low information
    #rtrain = rtrain.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)
    #rtest = rtest.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC'], axis=1)
    
    # Correcting categorical values that are ordinal
    filling_ordinal(rtrain)
    filling_ordinal(rtest)
    
    #Feature ingeneering
    feat_ing(rtrain)
    feat_ing(rtest)
    
    #One hot encoding
    rtrain=pd.get_dummies(rtrain)
    rtest=pd.get_dummies(rtest)
    
    # Update the training set
    rtrain=rtrain[rtest.columns]
    
    
    return(rtrain,rtest,y,testId)


# In[ ]:


X,Xtest,y,TestId=clean(train_copy,test_copy)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.33, random_state=77)


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
parameters= {'alpha':[0.0001,0.0002,0.0003,0.0004,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}

lasso=Lasso()
lasso_reg=GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)
lasso_reg.fit(X,y)

print('The best value of Alpha is: ',lasso_reg.best_params_,'neg_mean_squared_error',lasso_reg.best_score_)


# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import Lasso

best_alpha=0.0009
lasso = Lasso(alpha=best_alpha,max_iter=10000)
lasso.fit(X,y)


# In[ ]:


ytest=lasso.predict(Xtest)
#ytest_ridge=ridge.predict(Xtest)
#ytest=(ytest_ridge+ytest_lasso)/2


# In[ ]:



ytest=np.expm1(ytest)


# In[ ]:


#FI_lasso = pd.DataFrame({"Feature":X.columns, 'Importance':lasso.coef_})


# In[ ]:


#FI_lasso=FI_lasso.sort_values("Importance",ascending=False)


# In[ ]:


#import seaborn as sns
#sns.barplot(x='Importance', y='Feature', data=FI_lasso.head(10),color='b')


# In[ ]:


prediction=pd.DataFrame({'Id': TestId, 'SalePrice': ytest})


# In[ ]:


prediction


# In[ ]:


prediction.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




