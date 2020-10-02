#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn import ensemble
from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor


# In[ ]:


train=pd.read_csv('/kaggle/input/dasprodatathon/train.csv')
test=pd.read_csv('/kaggle/input/dasprodatathon/test.csv')


# In[ ]:


print('DataFrame Train')
display(train)


# In[ ]:


print("DataFrame Test")
display(test)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


#print first five rows of the dataframe
train.head()


# In[ ]:


print(train.info())
print('**'* 50)
print(test.info())


# In[ ]:


#Describe gives statistical information about numerical columns in the dataset
train.describe()


# **Data Visualization**

# In[ ]:


sns.distplot(train['Price'] , fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['Price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['Price'], plot=plt)
plt.show() 


# the price value is right skewed.

# In[ ]:


plt.figure(figsize=(30,8))
sns.heatmap(train.corr(),cmap='coolwarm',annot = True)
plt.show()


# we can see the most corelated parameters in numerical values above plotting. And we can pick these as features for our macine learning mode

# In[ ]:


#train['Price'].skew()
print("Skewness: %f" % train['Price'].skew())
print("Kurtosis: %f" % train['Price'].kurt())


# In[ ]:


plt.hist(train.Price, bins = 25)


# In[ ]:


#corr=train.corr()["Price"]
#corr[np.argsort(corr, axis=0)[::-1]]

correlations = train.corr()
correlations = correlations["Price"].sort_values(ascending=False)
features = correlations.index[1:6]
correlations


# In[ ]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


k = 10 #number of variables for heatmap
cols = train.corr().nlargest(k, 'Price')['Price'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# we can see the most corelated parameters in numerical values above plotting. And we can pick these as features for our macine learning model.

# In[ ]:


#scatterplot
sns.set()
cols = ['Price', 'Living Area', 'Grade', 'Above the Ground Area', 'Neighbors Living Area', 'Bathrooms', 'View', 'Basement Area', 'Latitude', 'Bedrooms']
sns.pairplot(train[cols], size = 2.5)
plt.show();


# In[ ]:


sns.distplot(train['Price'], color="r", kde=False)
plt.title("Distribution of Price")
plt.ylabel("Number of Occurences")
plt.xlabel("Price");


# In[ ]:


most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr


# In[ ]:


sns.jointplot(x=train['Living Area'], y=train['Price'], kind='reg')


# **Feature Engineering**

# In[ ]:


train_null = pd.isnull(train).sum()
test_null = pd.isnull(test).sum()

null = pd.concat([train_null, test_null], axis=1, keys=["Training", "Testing"])


# In[ ]:


null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #not as much missing values


# In[ ]:


null_many


# In[ ]:


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# In[ ]:


#missing data
total_test = test.isnull().sum().sort_values(ascending=False)
percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# In[ ]:


train=train[cols]


# In[ ]:


cols


# In[ ]:


train.columns


# *The primary concern here is to establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data. In this context, data standardization means converting data values to have mean of 0 and a standard deviation of 1.*

# In[ ]:


#skew_cols = ['Living Area', 'Total Area', 'Above the Ground Area',
 #      'Basement Area', 'Neighbors Living Area', 'Neighbors Total Area']


# In[ ]:


#test['rooms_ratio'] = test['Living Area'] / (test['Bedrooms'] + test['Bathrooms'])
#train['rooms_ratio'] = train['Living Area'] / (train['Bedrooms'] + train['Bathrooms'])


# In[ ]:


#for df in [train,test]:
 #   df['living_ratio'] = df['Living Area'] / df['Total Area']
  #  df['neighbors_ratio'] = df['Neighbors Living Area'] / df['Neighbors Total Area']
   # df['env_ratio'] = df['Living Area'] / df['Neighbors Living Area']


# In[ ]:


#train['per_price'] = train['Price']/train['Total Area']
#zipcode_price = train.groupby(['Zipcode'])['per_price'].agg({'mean','var'}).reset_index()
#train = pd.merge(train,zipcode_price,how='left',on='Zipcode')
#test = pd.merge(test,zipcode_price,how='left',on='Zipcode')
#train.drop('per_price', axis=1, inplace=True)

#for df in [train,test]:
 #   df['zipcode_mean'] = df['mean'] * df['Total Area']
  #  df['zipcode_var'] = df['var'] * df['Total Area']
   # del df['mean']; del df['var']


# In[ ]:


#threshold = 3
#z_score = np.abs(stats.zscore(train))

#train_wo = train[(z_score < threshold).all(axis=1)].copy()
#train_wo.head()


# train test split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Price', axis=1), train['Price'], test_size=0.2, random_state=3)


# In[ ]:


# we are going to scale to data
y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)

#from sklearn.impute import SimpleImputer
#my_imputer = SimpleImputer()
#data_with_imputed_values = my_imputer.fit_transform(original_data)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)


# In[ ]:


X_train


# In[ ]:


#x = train[['Living Area', 'Above the Ground Area','Neighbors Living Area','Bathrooms', 'Grade', 'Basement Area','Latitude','Longitude','Year Built','Condition','Bedrooms','Floors','Waterfront','View','Year Renovated','Neighbors Total Area','Total Area','Zipcode']]
#y = train['Price']


# In[ ]:


#x


# In[ ]:


id_test = test['ID']


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)
print(lr)


# In[ ]:


#y


# In[ ]:


#x, x_t, y, y_t = train_test_split(x,y, test_size=0.1, random_state=4)


# In[ ]:


#lr.fit(x,y)


# In[ ]:


print(lr.intercept_)


# In[ ]:


print(lr.coef_)


# In[ ]:


#lr.score(x,y)


# In[ ]:


#lr.score(x_t,y_t)


# In[ ]:





# In[ ]:


#id_test = test['ID']
#x_test = test[['Living Area', 'Above the Ground Area','Neighbors Living Area','Bathrooms', 'Grade', 'Basement Area','Latitude','Longitude','Year Built','Condition','Bedrooms','Floors','Waterfront','View','Year Renovated','Neighbors Total Area','Total Area','Zipcode']]


# In[ ]:


#y_pdc = lr.predict(x_test)
#display(y_pdc)


# In[ ]:


#est = ensemble.GradientBoostingRegressor(n_estimators=500,max_depth=4, random_state=8, min_samples_split = 10, learning_rate = 0.1,validation_fraction=0.1, loss = 'huber').fit(x,y)
#est.score(x,y)


# In[ ]:


#aw=est.score(x_t, y_t)
#aw= aw*100
#print(aw)


# In[ ]:


#y_pred = est.predict(x_test)
#display(y_pred)


# In[ ]:


#df_submit = pd.DataFrame()
#df_submit['ID'] = id_test
#df_submit['Price'] = y_pred


# In[ ]:





# In[ ]:


#df_submit.head()


# In[ ]:


#df_submit.to_csv('Submission_RIA.csv', index=False)


# In[ ]:


predictions = lr.predict(X_test)
predictions= predictions.reshape(-1,1)


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(predictions, label = 'predict')
plt.show()


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# **Gradient Boosting Regression**

# In[ ]:


from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


params = {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)


# In[ ]:


clf_pred=clf.predict(X_test)
clf_pred= clf_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,clf_pred, c= 'brown')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(clf_pred, label = 'predict')
plt.show()


# **Decision Tree Regression**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(random_state = 100)
dtreg.fit(X_train, y_train)


# In[ ]:


dtr_pred = dtreg.predict(X_test)
dtr_pred= dtr_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))
print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,dtr_pred,c='green')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# **Support Vector Machine Regression**

# In[ ]:


from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X_train, y_train)


# In[ ]:


svr_pred = svr.predict(X_test)
svr_pred= svr_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))
print('MSE:', metrics.mean_squared_error(y_test, svr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,svr_pred, c='red')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(svr_pred, label = 'predict')
plt.show()


# **Random Forest Regression**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 500, random_state = 0)
rfr.fit(X_train, y_train)


# In[ ]:


rfr_pred= rfr.predict(X_test)
rfr_pred = rfr_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))
print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,rfr_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(rfr_pred, label = 'predict')
plt.show()


# **LightGBM**
# 
# Light GBM is a gradient boosting framework that uses tree based learning algorithm.

# In[ ]:


import lightgbm as lgb


# In[ ]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=3000,
                            max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


model_lgb.fit(X_train, y_train)


# In[ ]:


lgb_pred = model_lgb.predict(X_test)
lgb_pred = lgb_pred.reshape(-1,1)


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))
print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))


# In[ ]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,lgb_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[ ]:


plt.figure(figsize=(16,8))
plt.plot(y_test,label ='Test')
plt.plot(lgb_pred, label = 'predict')
plt.show()


# **MODEL COMPARISON**
# 
# Dilihat dari error rate nya. semaakin kecil semakin akurat.

# In[ ]:


error_rate=np.array([metrics.mean_squared_error(y_test, predictions),metrics.mean_squared_error(y_test, clf_pred),metrics.mean_squared_error(y_test, dtr_pred),metrics.mean_squared_error(y_test, svr_pred),metrics.mean_squared_error(y_test, rfr_pred)])


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(error_rate)


# In[ ]:


a = pd.read_csv('/kaggle/input/dasprodatathon/test.csv')


# In[ ]:


#y_pdc=clf.predict(X_test)
#display(y_pdc)


# In[ ]:





# In[ ]:


test_id = a['ID']
a=pd.DataFrame(test_id, columns=['ID'])


# In[ ]:


test = sc_X.fit_transform(test)


# In[ ]:


test.shape


# In[ ]:


test_prediction_lgb=model_lgb.predict(X_test)

test_prediction_lgb=test_prediction_lgb.reshape(-1,1)


# In[ ]:


test_prediction_lgb


# In[ ]:


test_prediction_lgb =sc_y.inverse_transform(test_prediction_lgb)


# In[ ]:


test_prediction_lgb = pd.DataFrame(test_prediction_lgb, columns=['Price'])


# In[ ]:


test_prediction_lgb.head()


# In[ ]:


display(test_prediction_clf)


# In[ ]:


result = pd.concat([a,test_prediction_lgb], axis=1)


# In[ ]:


result.head()


# In[ ]:


result = result.fillna(method = "ffill" , axis = 0).fillna(0)


# In[ ]:


display(result)


# In[ ]:


result.to_csv('bismillah_submit.csv', index=False)

