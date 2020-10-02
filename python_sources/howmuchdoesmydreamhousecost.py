#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **DATA LOADING**

# In[ ]:


tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')


# **QUICK INSPECTION**

# In[ ]:


tr.info()


# In[ ]:


inspect_train = pd.DataFrame({'Dtype': tr.dtypes, 'Unique values': tr.nunique() ,
             'Number of Missing values': tr.isnull().sum() ,
              'Percentage Missing': (tr.isnull().sum() / len(tr)) * 100
             }).sort_values(by='Number of Missing values',ascending = False)
print('There are {} columns with no missing values'.format((inspect_train['Percentage Missing']==0.0).sum()))
inspect_train


# In[ ]:


inspect_test = pd.DataFrame({'Dtype': te.dtypes, 'Unique values': te.nunique() ,
             'Number of Missing values': te.isnull().sum() ,
              'Percentage Missing': (te.isnull().sum() / len(te)) * 100
             }).sort_values(by='Number of Missing values',ascending = False)
print('There are {} column with no missing values'.format((inspect_test['Percentage Missing']==0.0).sum()))
inspect_test


# There are 79 predictors.  Train set has 62 variables (that includes target and id) with no missing values, however test set has only 47 (that includes Id) with no missing values. 
# First interesting question. What should be the threshold t on percentage of missing values on a feature(predictor variable), to say that if missing value percentage > t, discard that predictor?

# In[ ]:


_ = plt.figure(figsize=[20,5])
_ = plt.plot(inspect_train['Percentage Missing'])
_ = plt.plot(inspect_test['Percentage Missing'],color='r')
_ = plt.legend(['train','test'])
_ = plt.grid()
_ = plt.title(' Missing value % in each of 79 predictors, plotted in descending order')
_ = plt.ylabel('Missing Percentage')
_ = plt.xlabel('Column index, when arranged according to the number of missing entries')


# The above plot gives a better idea on missing value percentage and it is helpful in intuitively deciding a threshold t. It looks like if t=10%, we lose only a smal number of predictors, i.e. 6 in both test and train set. Let us examine.

# In[ ]:


print('There are {} and {} columns with more than 90% missing values in test and train'.format((inspect_test['Percentage Missing']> 90.0).sum(),(inspect_train.drop('SalePrice')['Percentage Missing'] > 90.0).sum()))
print('There are {} and {} columns with more than 10% missing values in test and train'.format((inspect_test['Percentage Missing']> 10.0).sum(),(inspect_train.drop('SalePrice')['Percentage Missing'] > 10.0).sum()))
print('There are {} and {} columns with more than 50% missing values in test and train'.format((inspect_test['Percentage Missing']> 50.0).sum(),(inspect_train.drop('SalePrice')['Percentage Missing'] > 50.0).sum()))


# This is fine because the one extra in train set is due to target variable. Also note that 'Id' column is not a predictor.  In effect, there are equal number of variables of interest, if we choose this threshold. But there is no guarantee that in a new test set, 10% threshold will give 74 predictors! We should take care of it later if that happens.

# For the time being we will focus on 73 out of 79. To start with, we could even try to predict only with those variables which have no missing values in both train and test set. As first step, let's analyze the data bit more and understand these first 46 predictors.

# ![](http://)**Print the names of 46 predictors**

# In[ ]:


inspect_test.loc[(inspect_test['Percentage Missing']==0.0),:].index


# **DATA PREPARATION**

# In[ ]:


# Select data where there are <= 0% missing values
tr1 = tr.loc[:,inspect_test.loc[(inspect_test['Percentage Missing']==0.0),:].index]
tr1['SalePrice']=tr['SalePrice']
te1 = te.loc[:,inspect_test.loc[(inspect_test['Percentage Missing']==0.0),:].index]


# In[ ]:


categorical_variables = te1.drop('Id',axis=1).select_dtypes(exclude=['int64', 'float64', 'bool']).columns
numeric_variables = te1.drop('Id',axis=1).select_dtypes(include=['int64', 'float64']).columns
tr1.loc[:,numeric_variables].nunique()


# In[ ]:


te1.nunique()


# The year built or year remod by itself does not make sense. We can convert it into how many years old by subtracting by 2018. We should examine data to see if all data are valid. We should repeat on test set too. Yr sold has to be mostly converted to certain interval.

# In[ ]:


from datetime import date
def modify_data(data):
    data['YearBuilt'].apply(lambda x : np.nan if (date.today().year - x) < 0 else x) 
    data['YearRemodAdd'].apply(lambda x : np.nan if (date.today().year - x) < 0 else x) 
    data['age_of_house'] = date.today().year - data['YearBuilt'] 
    data['log_age'] = np.log(data.age_of_house+0.00001)
    data['time_since_remod'] = date.today().year - data['YearRemodAdd']
    data['log_remod'] = np.log(data.time_since_remod+0.00001)
    data.drop(['age_of_house','time_since_remod','YearBuilt','YearRemodAdd'],axis=1,inplace=True)
    return data
def standardize_data(data,num_var):
    epsilon=1e-8
    data.loc[:,num_var] = data.loc[:,num_var].transform(lambda x: (x - x.mean())/(x.std()+epsilon))
    return data


# In[ ]:


tr1 = modify_data(tr1)
te1 = modify_data(te1)


# In[ ]:


numeric_variables = list(te1.drop('Id',axis=1).select_dtypes(include=['int64', 'float64']).columns)
numeric_variables_tr = list(tr1.drop('Id',axis=1).select_dtypes(include=['int64', 'float64']).columns)


# In[ ]:


numeric_variables


# In[ ]:


tr1.loc[:,numeric_variables].nunique()


# In[ ]:


te1.loc[:,numeric_variables].nunique()


# Find Multicollinearity and remove those with above 0.95 correlation value

# In[ ]:


correlation_threshold = 0.95

corr_matrix = tr1.loc[:,numeric_variables_tr].corr()
# Extract the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Select the features with correlations above the threshold
# Need to use the absolute value
to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
print("There are "+str(len(to_drop))+" columns with correlations above certain threshold")


# In[ ]:


# Conversion to one hot
def conv_one_hot(data,categorical_variables) :
    #if(tr1.loc[:,var].dtype is object): 
    #tr1.loc[:,var].fillna('UNK',inplace=True)
    #    te1.loc[:,var].fillna('UNK',inplace=True)
    # Convert Categorical variables to dummies
    categorical_variables = data.select_dtypes(exclude=['int64', 'float64', 'bool']).columns
    cat_var = pd.get_dummies(data.loc[:,categorical_variables],drop_first=True)
    # Remove originals
    data = data.drop(categorical_variables,axis=1)
    data = pd.concat([data,cat_var],axis=1)
    #removing dulpicate columns - useful in case two variables are closely related.
    _, i = np.unique(data.columns, return_index=True)  #Always better to have
    data=data.iloc[:, i] 
    return data


# In[ ]:


tr1.shape,te1.shape


# In[ ]:


tr1 = conv_one_hot(tr1,categorical_variables)
te1 = conv_one_hot(te1,categorical_variables)


# In[ ]:


tr1.shape,te1.shape


# Note that after one hot conversion, the train set has 12 variables more than test set. It is because some values never appear in test set. They cause problem in prediction, hence they are removed from train set

# In[ ]:


Sprice = tr1.SalePrice
tr1.drop(list(set(tr1.columns)-set(te1.columns)),axis=1,inplace=True)
# Add target back
tr1['SalePrice'] = Sprice


# 1. **DATA VISUALIZATIONS**

# In[ ]:


_ = sns.distplot(tr1.SalePrice/1000)
_ = plt.title('Distribution of SalesPrice')
_ = plt.xlabel('SalesPrice in multiples of 1k dollars')
_ = plt.ylabel('Distribution')


# We would choose to visualize scatterplots of those with many unique values. They are the real continuous variables. Remaining are ordinal variables for which barplot is suitable

# In[ ]:


_ = plt.scatter(tr1.BedroomAbvGr,tr1.SalePrice)
print(tr1.groupby('BedroomAbvGr').agg({'SalePrice':['mean','std','min','max','count']}))


# In[ ]:


_ = sns.regplot(x='KitchenAbvGr',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs KitchenAbvGr')


# In[ ]:


_ = plt.plot(tr1.groupby('MoSold').SalePrice.mean())
_ = plt.title('SalePrice Vs Month')
tr1.groupby('MoSold').SalePrice.mean()


# There is good variation with respect to month, but Month is neither numeric nor ordinal variable. It is categorical by definition!
# Hence it is important to take care of this later.

# In[ ]:


fig = plt.figure(figsize=(20, 10))
_ = plt.grid()
_ = plt.subplot(3,3,1)
_ = sns.regplot(x='LotArea',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs Lot Area')
_ = plt.subplot(3,3,2)
_ = sns.regplot(x='ScreenPorch',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs ScreenPorch')
_ = plt.subplot(3,3,3)
_ = sns.regplot(x='EnclosedPorch',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs EnclosedPorch')
_ = plt.subplot(3,3,4)
_ = sns.regplot(x='GrLivArea',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs GrLivArea')
_ = plt.subplot(3,3,5)
_ = sns.regplot(x='OpenPorchSF',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs OpenPorchSF')
_ = plt.subplot(3,3,6)
_ = sns.regplot(x='WoodDeckSF',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs WoodDeckSF')
_ = plt.subplot(3,3,7)
_ = sns.regplot(x='1stFlrSF',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs 1stFlrSF')
_ = plt.subplot(3,3,8)
_ = sns.regplot(x='2ndFlrSF',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs 2ndFlrSF')
_ = plt.subplot(3,3,9)
_ = sns.regplot(x='log_age',y='SalePrice',data=tr1,fit_reg=False)
_ = plt.title('SalePrice Vs Log of Age of the House')


# We see quite a bit of linear relationships here. Especially, GrLivArea seems to be a strong predictor. Some of the predictors have lot of 0's. It is not invalid data, such houses may not have that provision (like ScreenPorch).

# In[ ]:


tr1 = conv_one_hot(tr1,['MoSold','YrSold'])
te1 = conv_one_hot(te1,['MoSold','YrSold'])
tr1 = standardize_data(tr1,numeric_variables) 
te1 = standardize_data(te1,numeric_variables)


# **PREDICTION**      
# Choice of Evaluation Metric:
# We would go with RMSE to measure the final model. To evaluate models, we can use several.

# In[ ]:


from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
def compute_rmse(y,ypred):
    mse = np.mean((y-ypred)*(y-ypred))
    rmse = np.sqrt(mse)
    return rmse
np.random.seed(13)


# In[ ]:


X = tr1.drop(['Id','SalePrice'],axis=1)
y = tr1['SalePrice']


# In[ ]:


Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.1,random_state=43)


# In[ ]:


# Print certain benchmarks
def print_measures(Xte,yte,ypred):
    print("Number of predictor variables: ",Xte.shape[1])
    rsq = r2_score(ypred,yte)
    print("R2 score on test set:  ",np.round(rsq,4))
    adj_rsq = 1-((1-rsq)*(Xte.shape[0]-1)/(Xte.shape[0]-Xte.shape[1]-1))
    print("Adjusted R2 score on test set: ",np.round(adj_rsq,4))
    print('RMSE on test set: ',np.round(compute_rmse(ypred,yte),4))


# In[ ]:


# Select those numeric variables which are really numeric and not ordinal.
select_variables = ['MiscVal',
 'ScreenPorch',
 '3SsnPorch',
 'EnclosedPorch',
 'OpenPorchSF',
 'WoodDeckSF',
 'GrLivArea',
 'LotArea',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'log_age',
 'log_remod']
lr = LinearRegression()
lr.fit(Xtr.loc[:,select_variables],ytr)
ypred = lr.predict(Xte.loc[:,select_variables])
print_measures(Xte.loc[:,select_variables],yte,ypred)
coeff=lr.coef_
intercept = lr.intercept_
coeffs_b= lr.coef_[np.argsort(abs(lr.coef_))[::-1]]
names_b = list(Xte.loc[:,select_variables].columns[np.argsort(abs(lr.coef_))[::-1]])
lrimp = pd.DataFrame(np.round(coeffs_b,3),index=names_b,columns=['Coeff value'])
_ = np.log(np.abs(lrimp)).plot.bar(color='purple')
_ = plt.title('Feature Importance (Linear Reg)')
_ = plt.ylabel('Coefficient value')
_ = plt.xlabel('Features')
lrimp


# In[ ]:


# First model using only 25 numeric variables. This model has neither converted nor used categorical variables.
lr = LinearRegression()
lr.fit(Xtr.loc[:,numeric_variables],ytr)
ypred = lr.predict(Xte.loc[:,numeric_variables])


# In[ ]:


print_measures(Xte.loc[:,numeric_variables],yte,ypred)


# In[ ]:


coeff=lr.coef_
intercept = lr.intercept_
coeffs_b= lr.coef_[np.argsort(abs(lr.coef_))[::-1]]
names_b = list(Xte.loc[:,numeric_variables].columns[np.argsort(abs(lr.coef_))[::-1]])
lrimp = pd.DataFrame(np.round(coeffs_b,3),index=names_b,columns=['Coeff value'])
_ = np.log(np.abs(lrimp)).plot.bar(color='purple')
_ = plt.title('Feature Importance (Linear Reg)')
_ = plt.ylabel('Coefficient value')
_ = plt.xlabel('Features')
lrimp


# Note that just the addition of 12 more ordinal predictors has increased both R^2 and adj R^2.

# In[ ]:


lr = LinearRegression()
lr.fit(Xtr,ytr)
ypred = lr.predict(Xte)
print_measures(Xte,yte,ypred)


# We can see that adding all variables, even though one hot encoded, does increase R^2, but decreases adjusted R^2 from 0.8 to -0.54.
# This means, the model with categorical variables, is not necessarily better than the model with just 25 numeric variables.
# But this last model will give a better position in leaderboard.

# In[ ]:


ysub = lr.predict(te1.drop('Id',axis=1))
te1 = te1.assign(SalePrice = ysub)
te1[['Id','SalePrice']].to_csv('aparna_housing_price.csv',index=False)


# [Why Encoding](https://datascience.stackexchange.com/questions/5226/strings-as-features-in-decision-tree-random-forest)
