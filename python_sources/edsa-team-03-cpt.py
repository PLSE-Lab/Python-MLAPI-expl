#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # House Price Prediction
# 
# Given a set of features for several houses including their prices. Predict the price of unseen houses.

# #### Import all relevant libraries

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#set the plots to use the default seaborn style
sns.set()


# In[ ]:


# Set the option to view all columns
pd.set_option('display.max_columns', None)


# ## Import the data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# ## Merge the two datasets for easy data manipulation

# *`Before we merge the two datasets together, lets first make sure that they are of the same shape. We will now add SalePrice to test_df and call head() to see the changes`*

# test_df['SalePrice'] = 0

# In[ ]:


test_df.head()


# In[ ]:


df = train_df.append(test_df)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# ### Let us view the information on our train dataset

# In[ ]:


print(train_df.info())


# ## Investigate all columns with missing values of over 10% of the data

# In[ ]:


## Get a list of columns except for the SalePrice column
columns = list(df.columns.values)
columns.remove('SalePrice')


# In[ ]:


#get the length of combined dataframe
df_N = df.shape[0]


# In[ ]:


#get a list of all columns with atleast 10% missing values
missing_val_cols ={}
for col in columns:
    col_empties_N = df[col][df[col].isnull()].shape[0]
    empties_perc = (col_empties_N/df_N) * 100
    if empties_perc > 10:
        missing_val_cols[col] = empties_perc


# In[ ]:


pd.Series(missing_val_cols).plot.bar()


# ## Fill the values accordingly
# 
# 
# From the above investigate the nature of the missing values. What does the presence of missing values mean for this dataset?
# Careful inspection of the above coupled with the instructions for the dataset we have come to realise that NA values have been encorded as NaN hence we will go ahead and convert them into the string literal "None". Plus looking at other numerical variables where it is believed that the feature does not exist hence unmeasurable, zero (0) hence been used in that case. We will go ahead and do it for the "LofFrontage" column

# In[ ]:


df['LotFrontage'][df['LotFrontage'].isnull()] = 0


# In[ ]:


df['Alley'][df['Alley'].isnull()] = 'None'


# In[ ]:


df['FireplaceQu'][df['FireplaceQu'].isnull()] = 'None'


# In[ ]:


df.PoolQC[df.PoolQC.isnull()] = 'None'


# In[ ]:


df['Fence'][df['Fence'].isnull()] = 'None'


# In[ ]:


df['MiscFeature'][df['MiscFeature'].isnull()] = 'None'


# In[ ]:


#let us see how the dataframe now looks like
df.head()


# In[ ]:


df[df.dtypes[df.dtypes == 'int64'].index].fillna(0,inplace=True)


# In[ ]:


#use forward fill to fill the missing values
df.fillna(method='ffill',inplace=True)


# In[ ]:


df.tail()


# ## Create Dummy Variables
# 
# 
# Since most machine learning models/algorithms require that the input data be numerical. We are going to go through each column that contains categorical data and create dummy columns for each.

# In[ ]:


from collections import defaultdict


# In[ ]:


object_cols = [] #container to store all column names where the data type is "object"
col_dummies = [] #container to store all dataframes created from pandas get_dummies method

for col in df.columns:
    if str(df[col].dtypes) == 'object':
        object_cols.append(col)
        col_dummy = pd.get_dummies(df[['Id',col]],drop_first=True,prefix=col)
        col_dummies.append(col_dummy)  


# In[ ]:


#drop off all columns where the data
df_clean = df.drop(object_cols,axis=1)
#now append each dataframe from above to the left of our original dataframe
for index,col in enumerate(object_cols):
    df_clean= pd.merge(df_clean,col_dummies[index],on='Id') 
df_clean.set_index('Id',inplace=True)


# ### Investigate the relationships between predictors and the response
# 
# After running OLS on the training dataset, we found out that only two predictors have a correlation of over 0.7. We will go ahead plot the relationship between the two predictors and response, respectively

# In[ ]:


train_df.plot.scatter(x='OverallQual',y='SalePrice')


# In[ ]:


plt.figure(figsize=(8,7))
sns.boxplot(y='SalePrice',x='OverallQual',data=train_df)


# By plotting a boxplot for OverallQual against SalePrice we can easily predict the price of a house based on its overall quality rating. The prices do overlap but the average price for each rating is easily distinguished from the rest.

# In[ ]:


train_df.plot.scatter(x='GrLivArea',y='SalePrice')


# ### Engineer Features based on the above
# 
# It looks like OverallQual takes on a quadratic shape. So lets go ahead and transform it. In actual fact, after taking the sqaure of OverallQual it's correlation to the sale price increases by two units i.e from 79% to 81%. We can also see that there are outliers on GrLivArea. We will need to deal with these later on.

# In[ ]:


df_clean['OverallQual^2'] = df_clean['OverallQual'] ** 2
df_clean.drop('OverallQual',inplace=True,axis=1)


# In[ ]:


df_clean.head()


# It looks like some columns are closely related. Lets go ahead and merge them together.

# In[ ]:


#df_clean = df_clean.drop(df_clean[(df_clean['GrLivArea']>4000)].index)# Summarize features
df_clean['TotalSF'] = df_clean['1stFlrSF'] + df_clean['2ndFlrSF'] + df_clean['TotalBsmtSF']
df_clean = df_clean.drop(['1stFlrSF','2ndFlrSF','TotalBsmtSF'], axis=1)

df_clean['TotalArea'] = df_clean['LotFrontage'] + df_clean['LotArea']
df_clean = df_clean.drop(['LotFrontage','LotArea'], axis=1)

df_clean['BSF'] = df_clean['BsmtFinSF1'] + df_clean['BsmtFinSF2']
df_clean = df_clean.drop(['BsmtFinSF1','BsmtFinSF2'], axis=1)

df_clean['BsmtBath'] = df_clean['BsmtFullBath'] + (0.5 * df_clean['BsmtHalfBath'])
df_clean = df_clean.drop(['BsmtFullBath','BsmtHalfBath'], axis=1)

df_clean['Bath'] = df_clean['FullBath'] + (0.5 * df_clean['HalfBath'])
df_clean = df_clean.drop(['FullBath','HalfBath'], axis=1)


# In[ ]:


#get the correlation values against each column
train_length = train_df.shape[0]
corrs = df_clean.iloc[:train_length].corr()


# In[ ]:


#convert them into a dataframe
corrs_df = pd.DataFrame(corrs.SalePrice)


# In[ ]:


#pick only those whose correlation with the sale price is over 0.5
contenders = corrs_df[corrs_df.SalePrice > 0.5]
contenders.drop('SalePrice',inplace=True)


# In[ ]:


#reset the index of the dataframe so that we may be able to access the columns names
contenders.reset_index(inplace=True)


# In[ ]:


#rename the new column for easy acces
contenders.columns = ['Predictor','Correlation']


# In[ ]:


#show the correlations in descending order
contenders.sort_values('Correlation',ascending=False)


# ### Look out for any correlation between contenders variables

# In[ ]:


corrs.loc[contenders.Predictor[:-1]][contenders.Predictor[:-1]]


# In[ ]:


to_plot = corrs.loc[contenders.Predictor][contenders.Predictor]
plt.figure(figsize=(10,10))
g = sns.heatmap(to_plot,annot=True,cmap="RdYlGn")


# > As expected, GarageCars and GarageArea are correlated
# 
# > So is, 1stFlrSF and TotalBsmtSF
# 
# > But 1stFlrSF < TotalBsmtSF
# 
# > and GarageCars > GarageArea

# #### Drop the least correlated variables to the sale price from above.

# In[ ]:


to_drop = contenders.iloc[[3,8,5]]
to_drop


# In[ ]:


contenders = contenders.drop([3,8,5])
contenders.sort_values('Correlation',ascending=False)


# In[ ]:


corrs.loc['SalePrice'][contenders.Predictor].plot.bar()
plt.title('Top 6 Predictors Correlation to SalePrice')
plt.ylabel('Correlation')


# ## Split the Data into train, test and validation sets

# In[ ]:


#get the length of the original train_df dataframe
train_length = train_df.shape[0]


# In[ ]:


#make X by taking only the rows from train_df besides the SalePrice column
X = df_clean.iloc[:train_length].drop('SalePrice',axis=1)


# In[ ]:


X.head()


# In[ ]:


#make y by taking all the rows from train_df but only the SalePrice column
y = df_clean.iloc[:train_length]['SalePrice']


# 

# In[ ]:


#use the remaining data for testing the final model
X_validation = df_clean.iloc[train_length:].drop('SalePrice',axis=1)


# In[ ]:


X_validation.head()


# ### Removing outliers from our train set
# 
# Now that we have performed all necessary taks, it safe to remove all outliers identified by our GrLivArea column

# In[ ]:


idx_to_drop = X[(X['GrLivArea']>4000)].index
X.drop(X.loc[idx_to_drop].index,inplace=True)
y.drop(y.loc[idx_to_drop].index,inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=45)


# ## Create A Scaler Object
# 
# Since the data is measured in different metrices, we will need to normalise the data. To avoid any bias towards features with higher values. This will also help in making sure that the variance in our dataset is minimal. Thanks to scikits learn family of libraries, we have the right tool to get the job done.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# ## Create methods/functions to reuse
# 
# 
# Since we will be trying out different models to see which performs best. To avoid repeatition of code, we must create functions which we will call to perform these repeatative tasks.

# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score ,mean_squared_log_error


# In[ ]:


def get_model_scores(y_pred,y_test):
    """
        Calculates and returns the mean squared error,r squared score and mean squared log error of the given input
        
        input: y_pred array-like. Prediction values of a dataset
               y_test array-like. Actual values of a dataset
               
       output: tuple in the form of (mean squared error,r squared,mean squared log error) scores
    """
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    msle = mean_squared_log_error(y_test,y_pred)
    return mse,r2,msle


# In[ ]:


def print_model_score(y_pred,y_test):
    """
        Prints the mse, r-squared and msle score of the inputs
    
        input: y_pred array-like items of the predicted values
               y_test array-like items of the actual values   
    """
    test_mse,test_r2,test_log = get_model_scores(y_pred,y_test)
    print('Test MSE: {}'.format(test_mse))
    print('Test R-Squared Score: {}'.format(test_r2))
    print('Test Mean-Squared Log Error: {}'.format(test_log))


# In[ ]:


def plot_actual_preds(actuals,preds):
    """
        Plots the actual house prices overlayed on the predicted house prices.
        
        Input: actuals array-like values of target variable
               preds array-like predicted values of the target variable
    """
    plt.plot(actuals,linestyle=None,linewidth=0,marker='o',label='Actual Values',alpha=0.5)
    plt.plot(preds,color='red',linestyle=None,linewidth=0,marker='o',
         label='Predictions',alpha=0.2)
    plt.ylabel('Sale Price')
    plt.legend(loc=(1,1))
    plt.show()


# In[ ]:


def get_scalers(to_scale):
    return scaler.fit_transform(to_scale)


# ## Train the models

# In[ ]:


model_performance ={}


# ### Linear Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


linear_model = LinearRegression()


# In[ ]:


linear_model.fit(scaler.fit_transform(X_train[contenders.Predictor]),np.log(y_train))


# In[ ]:


y_pred = linear_model.predict(scaler.transform(X_test[contenders.Predictor]))


# In[ ]:


model_performance['Linear'] = get_model_scores(np.exp(y_pred),y_test)
print_model_score(y_test.values,np.exp(y_pred))


# In[ ]:


plot_actual_preds(y_test.values,np.exp(y_pred))


# ## Lasso Model

# In[ ]:


from sklearn.linear_model import LassoCV


# In[ ]:


lasso_model = LassoCV(cv=3)


# In[ ]:


lasso_model.fit(scaler.fit_transform(X_train),np.log(y_train))


# In[ ]:


y_pred = lasso_model.predict(scaler.transform(X_test))


# In[ ]:


model_performance['Lasso'] = get_model_scores(np.exp(y_pred),y_test)
print_model_score(np.exp(y_pred),y_test)


# ## Ridge Model

# In[ ]:


from sklearn.linear_model import RidgeCV


# In[ ]:


ridge_model = RidgeCV(cv=3)


# In[ ]:


ridge_model.fit(get_scalers(X_train),np.log(y_train))


# In[ ]:


y_pred = ridge_model.predict(get_scalers(X_test))


# In[ ]:


model_performance['Ridge'] = get_model_scores(np.exp(y_pred),y_test)
print_model_score(np.exp(y_pred),y_test)


# In[ ]:


plot_actual_preds(y_test.values,np.exp(y_pred))


# ## Random Forest Tree Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf_model = RandomForestRegressor(n_estimators=200,min_samples_leaf=3
                              ,max_features=0.5,warm_start=True,
                              bootstrap=False,random_state=123,
                                  )


# In[ ]:


rf_model.fit(scaler.fit_transform(X_train),np.log(y_train))


# In[ ]:


y_pred = rf_model.predict(scaler.fit_transform(X_test))


# In[ ]:


model_performance['RandomTree'] =  get_model_scores(np.exp(y_pred),y_test)
print_model_score(np.exp(y_pred),y_test)


# In[ ]:


rf_fi =pd.DataFrame(rf_model.feature_importances_,index=X_train.columns,columns=['Feature Importance'])
rf_fi = rf_fi[rf_fi['Feature Importance'] > 0].sort_values('Feature Importance',ascending=False)*100


# In[ ]:


#check how much each feature from our selected features, actually contributes
rf_fi.loc[contenders.Predictor].plot.bar()
plt.title('Top 6 Predictors')
plt.ylabel('Contribution (%)')


# In[ ]:


rf_fi.loc[contenders.Predictor].sum().plot.bar()
plt.title('Top 6 Predictors Total Contribution')
plt.ylabel('Contribution (%)')


# In[ ]:


rf_fi.head(13).plot.bar()
plt.title('Top 16 Predictors')
plt.ylabel('Total Contribution (%)')


# In[ ]:


plot_actual_preds(y_test.values,np.exp(y_pred))


# ## Extreme Gradient Boosting Model

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


xgb_model = GradientBoostingRegressor(alpha=0.95, criterion='friedman_mse',
             learning_rate=0.01, loss='huber',max_features='sqrt'
             ,min_samples_leaf=10,min_samples_split=10,n_estimators=3000,
             random_state=None, subsample=0.4)


# In[ ]:


xgb_model.fit(scaler.fit_transform(X_train),np.log(y_train))


# In[ ]:


y_pred = xgb_model.predict(scaler.fit_transform(X_test))


# In[ ]:


model_performance['XGB'] = get_model_scores(np.exp(y_pred),y_test)
print_model_score(np.exp(y_pred),y_test)


# In[ ]:


plot_actual_preds(y_test.values,np.exp(y_pred))


# ## Stacked Regressor Model

# In[ ]:


def blended_prediction(X):
    return ((0.45 * xgb_model.predict(X)) +             (0.25 * lasso_model.predict(X)) +             (0.25 * ridge_model.predict(X)) +            (0.05 * rf_model.predict(X))
            )


# In[ ]:


y_pred = blended_prediction(scaler.fit_transform(X_test))


# In[ ]:


model_performance['Stacked'] =get_model_scores(np.exp(y_pred),y_test)
print_model_score(np.exp(y_pred),y_test)


# In[ ]:


plot_actual_preds(y_test.values,np.exp(y_pred))


# # Select the best performing model

# In[ ]:


mp_array_dic =  {model:np.array(perf) for model,perf in model_performance.items()}
perf_df = pd.DataFrame(mp_array_dic).transpose()
perf_df.columns = ['MSE','R-Squared','MSLE']


# In[ ]:


perf_df[['MSE']].plot.bar(legend=None)
plt.title('Mean Sqaured Error per Model')
plt.ylabel("MSE (100 Million)")


# In[ ]:


perf_df[['R-Squared']].plot.bar(legend=None)
plt.title('R-Sqaured Score per Model')
plt.ylabel("Score")


# In[ ]:


perf_df[['MSLE']].plot.bar(legend=None)
plt.title('Mean Squared Log Error per Model')
plt.ylabel("Mean Squared Log Error")


# From the above options, it appears that *`Lasso`, `Ridge`* and *`XGB`* are the prime candidates. But combining the models together and taking a weighted contribution for each we get a better model. So lets go ahead and predict with the new model.

# ## Save results into the submission file

# In[ ]:


y_pred = blended_prediction(scaler.fit_transform(X_validation))
validation_df = X_validation.copy()
validation_df['SalePrice'] = np.exp(y_pred)
validation_df['SalePrice'].tail()


# In[ ]:


validation_df[['SalePrice']].tail()


# In[ ]:


validation_df[['SalePrice']].to_csv('submission.csv')

