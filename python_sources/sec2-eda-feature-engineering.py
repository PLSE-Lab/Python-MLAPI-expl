#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In the previous section, we have cleaned whole dataset(train and test) while maintained size and shape of data same as original.(Refer:https://www.kaggle.com/lajari/sec1-tedious-data-cleaning). We are going to use this cleaned data for further analysis and feature engineering. 
# 
# Our main focus in this section, will be on EDA, feature engineering and feature selection for numerical features only. At every step of our data analysis we are evaluating one test model to analyse the impact of our engineered data on overall performance in terms RMSE. This test model is nothing but just simple linear regressor. 
# 

# ### Load Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('max_rows',100,'max_columns',100)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ### Load Cleaned Data

# In[ ]:


train = pd.read_csv('/kaggle/input/sec1-tedious-data-cleaning/ctrain.csv')
test = pd.read_csv('/kaggle/input/sec1-tedious-data-cleaning/ctest.csv')

nulltrain = train.isnull().sum()
print(nulltrain[nulltrain>0])

nulltest = test.isnull().sum()
print(nulltest[nulltest>0])


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


catcols = train.select_dtypes(object).columns
numcols = train.select_dtypes(np.number).columns

print('Total categorical columns:',len(catcols),'\nTotal numerical columns:', len(numcols))


# In[ ]:


## Useful functions


def outlier_detector(df,feature):
    """
    Detect rows which contains negative/positive outlier in any feature columns
    
    """
    rows = (((df[feature] - df[feature].mean()) > (3*df[feature].std())) | ((df[feature] - df[feature].mean()) < (-3*df[feature].std()))).any(axis=1)
    return rows

    

def test_model(data,features,target):
    """
    Evaluate RMSE for simple Linear Regression model with given features and target
    
    """
    
    all_X = data[features].copy()
    
    scaler = StandardScaler()
    all_X = scaler.fit_transform(all_X)
    
    all_y = data[target].copy()
    
    X_train,X_test,y_train,y_test = train_test_split(all_X,all_y, test_size=0.3, random_state = 0,shuffle=True)
    
   
    model = LinearRegression().fit(X_train,y_train)
    prediction = model.predict(X_test)
    y_test = np.exp(y_test) - 1
    prediction =np.exp(prediction) - 1
    
    mse = mean_squared_error(y_test,prediction)
    rmse = np.sqrt(mse)
    
    print("Result of test model\n\nFeatures:",list(features),"\n\nTarget:",target,"\n\nRMSE:{:.4f}".format(rmse))

    
    


# ## Analyse SalePrice:
# Target column is right skewed and may contain some outliers. If we use parametric model for regression then it is important to normalize the data as it is underlaying assumption for most linear models. But if we use non parametric model such as tree based regressor it doesn't require.

# In[ ]:


# SalePrice Analysis

fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
logform = np.log1p(train['SalePrice'])
sqrtform =np.sqrt(train['SalePrice'])

sns.distplot(train['SalePrice'], bins =500,kde=False,ax=ax1).set_title('Skew: {:.2f}'.format(train['SalePrice'].skew()))
sns.distplot(logform,bins = 500,kde=False,ax=ax2).set_title('Skew: {:.2f}'.format(logform.skew()))
sns.distplot(sqrtform, bins= 500, kde=False, ax=ax3).set_title('Skew: {:.2f}'.format(sqrtform.skew()))


# In[ ]:


# We would chose log version of SalePrice as our target
train['LogPrice'] = logform

# Detect outliers in target column
outliers = outlier_detector(train,['LogPrice'])
print('Number of outliers:',outliers.sum())

# Remove Outliers
train = train[~outliers]


# In[ ]:


# Test model with all available features and newly created target variable 'LogPrice'
features = train.select_dtypes(np.number).columns.drop(['SalePrice','LogPrice'])
test_model(train,features,'LogPrice')


# ## EDA and Feature Engneering
# 
# Under this subsection, we will follow below steps 
# 1. Collinearity Analysis
# 2. Outlier Removal
# 3. Feature Engineering
# 
# ### Analysis for collinearity:
# 
# Collinear features affect the intepretability of machine learning model as well as it increases the complexity. Therefore, it is safe to remove them as it would not affect much on final prediction accuracy. In this context, correlation> 0.8 between two features is considered as collinear features.(Refer this link for more detail: https://medium.com/future-vision/collinearity-what-it-means-why-its-bad-and-how-does-it-affect-other-models-94e1db984168). We would analyze their role carefully before decide on their removal.

# In[ ]:


cormat = train.corr()
cormat.style.applymap(lambda x: 'background-color : yellow' if (x>0.8) & (x!=1) else '')


# In[ ]:


# We are definitely keeping feature which has high correlation with target like OverallQual.
# tuple list with very high correlation. 
tuplist = [('1stFlrSF','TotalBsmtSF') ,('TotRmsAbvGrd','GrLivArea'),('GarageArea','GarageCars')]

fig,axs = plt.subplots(1,3,figsize=(15,5))
axs = axs.flatten()
for i,tup in enumerate(tuplist):
    sns.scatterplot(train[tup[0]],train[tup[1]], ax = axs[i])
    plt.tight_layout()


# We found some interesting insight:
# 1. In some houses, basement is bigger than 1st Floor which is very less probability among all properties.(Refer:1st plot).
# 2. ratio of ground living area to total room above ground is very high for some house which could be possible if rooms are bigger than usual. This important information need to be preserved. So we generate ** new feature 'RoomSize'** (Refer: 2nd plot and boxplot below).
# 3. Garage capacity increases with Garage Area (Refer: 3rd Plot)
# 
# 
# In short we will remove collinear features ['TotRmsAbvGrd','GarageCars','TotalBsmtSF'] but after feature engineering stage.

# In[ ]:


# New Feature from above analysis is 'Roomsize'
train['RoomSize'] = train['GrLivArea']/train['TotRmsAbvGrd']
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
sns.boxplot(train[tuplist[1][0]],train[tuplist[1][1]],ax=ax1)
sns.boxplot(train[tuplist[1][0]],train['LogPrice'],ax=ax2)
sns.scatterplot(train['RoomSize'],train['LogPrice'],ax=ax3).set_title('corr:{:.2f}'.format(train['RoomSize'].corr(train['LogPrice'])))


# ### Outlier Removal:
# 
# There are many features which has significant correlation with target variable. As well as there are many data points that could possibly outlier in some pairplot in which Y-axis is SalePrice .For e.g. Unusual SalePrice for OverallCond 2 in following scatterpolt(2nd Row,2nd column). Our intutition is , It could be influence of other features which made SalePrice so unsusual. So It would be harmful if we remove such outlier. Instead, we will remove outliers which shows unusual behavior of feature. (E.g. Lot Area > 100000). 

# In[ ]:


# Visualizing outliers in scatterplot of feature vs target for first 20 numerical columns.
fig, axs = plt.subplots(5,4,figsize=(20,20))
axs = axs.flatten()
for i,col in enumerate(numcols[:20]):
    sns.scatterplot(train[col],train['LogPrice'], ax=axs[i])
    plt.tight_layout()


# In[ ]:


# First set of outlier based first 20 numerical columns

outlierrows1 = (train['LotFrontage']>200) | (train['LotArea']>100000) | (train['MasVnrArea']>1200) | (train['BsmtFinSF1'] > 3000) |                 (train['BsmtFinSF2'] > 1200) | (train['TotalBsmtSF'] > 4200) | (train['1stFlrSF'] >3500) | (train['GrLivArea'] >4000) |                 (train['BsmtFullBath'] > 2) | (train['BsmtHalfBath']> 1) 


# In[ ]:


# Visualizing outliers in scatterplot of feature vs target for remaining numerical columns.
fig, axs = plt.subplots(5,4,figsize=(20,20))
axs = axs.flatten()
for i,col in enumerate(numcols[20:]):
    sns.scatterplot(train[col],train['LogPrice'], ax=axs[i])
    plt.tight_layout()


# In[ ]:


# Second set of outliers based on remaining set of columns

outlierrows2 = (train['BedroomAbvGr'] > 7 ) |(train['KitchenAbvGr'] < 1 ) | (train['KitchenAbvGr'] > 2 ) |                (train['TotRmsAbvGrd'] > 12 ) | (train['WoodDeckSF'] > 800 ) | (train['OpenPorchSF'] > 450) | (train['EnclosedPorch'] > 400) |                (train['3SsnPorch'] > 350 ) | (train['MiscVal'] > 4000 ) 

# Remove outliers based on features
train = train[~(outlierrows1 | outlierrows2)]

train.shape
                                                                                                               


# In[ ]:


# Test model with data without outliers. Also remove 'Id' column as it has not much information and may affect interpretability of model.
features = train[numcols].columns.drop(['SalePrice','Id'])
test_model(train,features,'LogPrice')


# Our RMSE score improved significantly by outlier removal step. It has decresed from 157419.8091 to 21054.6703 which is remarkable improvement. It itself proved that unusual behavior of feature had big impact on model parameter estimation (in our case coefficient of linear model).Removing such outliers leads to actual estimate of model parameters.

# ### Feature Engineering 
# 
# We have created set of relevant feature and analyse them together for generating new features

# In[ ]:


numcols


# In[ ]:


# Let's make set of relevant features and analyse them together
s1 = ['LotFrontage', 'LotArea']
s2 = ['YearBuilt', 'YearRemodAdd','MoSold', 'YrSold']
s3 = ['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
s4 = ['1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea']
s5 = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath','HalfBath']
s6 = ['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']
s7 = ['GarageYrBlt', 'GarageCars', 'GarageArea']
s8 = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']


# In[ ]:


# Feature Engineering using set1 ['LotFrontage', 'LotArea']

# Assuming lot shape is approximately rectangle. We can develop following feature
train['LotPerimeter'] = 2*((train['LotArea']/train['LotFrontage'])+train['LotFrontage'])
s1.extend(['LotPerimeter'])
sns.pairplot(train[s1])


# In[ ]:


# Feature Engineering using set 2 ['YearBuilt', 'YearRemodAdd','MoSold', 'YrSold']

train['YrMoSold'] = train['YrSold']+0.01*train['MoSold']
train['Age'] = train['YrMoSold'] - train['YearBuilt']
train['IsRemod'] = train['YearBuilt'] != train['YearRemodAdd'] 
train['IsRemod'] = train['IsRemod'].map({True:1,False:0})       # This feature would be useful if we use tree based prediction method

s2.extend(['YrMoSold','Age','IsRemod'])
sns.pairplot(train[s2])


# In[ ]:


# Feature Engineering using set 3 ['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
train['TotFinBsmt'] = train['BsmtFinSF1'] + train['BsmtFinSF2']
train['IsBsmt'] = train['TotalBsmtSF'] != 0        # Useful in tree based modelling
train['IsBsmt'] = train['IsBsmt'].map({True:1,False:0})

s3.extend(['TotFinBsmt','IsBsmt'])
sns.pairplot(train[s3])


# In[ ]:


# Feature Engineering using Set 5 ['BsmtFullBath', 'FullBath','BsmtHalfBath', 'HalfBath']
train['TotFullBath'] = train['BsmtFullBath']+train['FullBath']
train['TotHalfBath'] = train['BsmtHalfBath'] +train['HalfBath']

# Feature Engineering using set 6 ['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']
# 'BedroomAbvGr'  is bedroom above grade.  We are not sure about what is grade here. Let's skip this step.

# Feature Engineering using set 7 ['GarageYrBlt', 'GarageCars', 'GarageArea'] - NA

# Feature Engineering using Set 8
train['TotPorch'] = train['OpenPorchSF']+train['EnclosedPorch']+train['3SsnPorch']+train['ScreenPorch']


# In[ ]:


# test model with engineered features along with original
dropcols = ['TotRmsAbvGrd','GarageCars','TotalBsmtSF','SalePrice','LogPrice','Id']
features = train.select_dtypes(np.float64).columns.drop(dropcols)
test_model(train,features,'LogPrice')


# It doesn'nt show any improvement in model performance. It could be possible that we have developed too many features. Therefore, We apply feature selection on further step to select most important features for linear regression model.

# In[ ]:


len(features)


# ## Feature Selection:
# 
# We will apply recursive feature elimination approach. This approach select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained through a coef_ attribute in our case. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

# In[ ]:


from sklearn.feature_selection import RFECV

scaler = StandardScaler()
all_X = train[features]
all_X = scaler.fit_transform(all_X)
all_y = train['LogPrice'].copy()

selector = RFECV(LinearRegression(), step=1, cv=3,scoring = 'neg_mean_squared_error')
selector = selector.fit(all_X,all_y)

selected_features = features[selector.support_]

test_model(train,selected_features,'LogPrice')
plt.plot(range(1, len(selector.grid_scores_) + 1), -selector.grid_scores_)
plt.xlabel('Number of features selected')
plt.ylabel('Cross Validation Score (MSE)')


# In[ ]:


plt.figure(figsize=(5,10))

print('number of selected features',selector.n_features_)
plt.barh(y=selected_features, width = selector.estimator_.coef_)


# RFE selected 38 features out of 41. It helps to reduce RMSE further to 20577. It means our assumption of having too many features in the model was correct. In above bar plot we have observed negative regression coefficient for positively correlated features. It's because of marginal effect in regression. 
# 
# We able to grab ranking position within **top 10%** on test data just by using EDA and FE for numerical features and with simplistic model. In the next section, we will work on categorical features as well as model selection strategy. We will store all the original and newly genearated features for furhter step.

# In[ ]:


def transform_feature(df):
    df['RoomSize'] = df['GrLivArea']/df['TotRmsAbvGrd']
    df['LotPerimeter'] = 2*((df['LotArea']/df['LotFrontage']) + df['LotFrontage'])
    df['YrMoSold'] = df['YrSold']+0.01*df['MoSold']
    df['Age'] = df['YrMoSold'] - df['YearBuilt']
    df['IsRemod'] = df['YearBuilt'] != df['YearRemodAdd'] 
    df['IsRemod'] = df['IsRemod'].map({True:1,False:0}) 
    df['TotFinBsmt'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df['IsBsmt'] = df['TotalBsmtSF'] != 0                    # Useful in tree based modelling
    df['IsBsmt'] = df['IsBsmt'].map({True:1,False:0})
    df['TotFullBath'] = df['BsmtFullBath']+df['FullBath']
    df['TotHalfBath'] = df['BsmtHalfBath'] +df['HalfBath']
    df['TotPorch'] = df['OpenPorchSF']+df['EnclosedPorch']+df['3SsnPorch']+df['ScreenPorch']
    
    return df

    


# In[ ]:


holdout = transform_feature(test)

scaled_df = pd.DataFrame(scaler.transform(holdout[features]),columns = features)

pred = selector.predict(scaled_df)

pred =np.exp(pred) - 1

submission_df = pd.DataFrame({'Id':holdout['Id'].astype(int),'SalePrice':pred})
submission_df.to_csv('submission.csv',index= False)


# In[ ]:


train.to_csv('engineered_train.csv',index=False)
holdout.to_csv('engineered_test.csv',index=False)

