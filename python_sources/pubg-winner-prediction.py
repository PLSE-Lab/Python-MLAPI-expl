#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/train_V2.csv')
datanew = data.copy()
data2 = pd.read_csv('../input/test_V2.csv')


# In[ ]:


data.head()


# In[ ]:


#Taking a sample of 50000 datapoints as dataset is very large
data = data.iloc[:50000,:]


# In[ ]:


#Checking for null values in the dataset
data.isnull().sum()


# In[ ]:


data1 = data.copy()


# In[ ]:


#Dropping Irrelevant attributes
data1.drop(['Id','groupId','matchId','matchType'],inplace=True,axis=1)
data2.drop(['groupId','matchId','matchType'],inplace=True,axis=1)


# In[ ]:


#Heatmap of correlation matrix
f,ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(data1.corr(),annot=True,fmt= '.1f',linewidths=.5)
plt.show()


# In[ ]:


#Top 5 attributes correlated to winPlacePerc
m = data1.corr()['winPlacePerc']
m.sort_values(ascending=False).head(6)


# In[ ]:


#Univariate Analysis
#1. matchType
f,ax = plt.subplots(figsize=(15,8))
sns.barplot(data.matchType.value_counts().values,data.matchType.value_counts().index)
plt.xlabel('Count')
plt.ylabel('matchType')
plt.show()


# In[ ]:


#Number of enemy players this player damaged that were killed by teammates.
#Number of assists greater than 0
d = data1.assists.value_counts()
d = d[d.index>0]


# In[ ]:


f,ax = plt.subplots(figsize=(15,5))
ax = sns.barplot(d.index,d.values,palette='rainbow')
plt.xlabel('Assists')
plt.ylabel('Count')
plt.show()


# In[ ]:


#Number of enemy players knocked.
#Maximum number of enemy players knocked by a player ranges between 0 and 5
f,ax = plt.subplots(figsize=(15,4))
ax = sns.distplot(data1.DBNOs,color='green')
plt.show()


# In[ ]:


#Duration of match in seconds.
#Most of the players play a match for around 1200 sec to 1800 sec
f,ax = plt.subplots(figsize=(15,4))
ax = sns.distplot(data1.matchDuration,color='green')
plt.show()


# In[ ]:


#Number of enemy players killed.
d = data1.kills.value_counts()
d = d[d.index>0]


# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
ax = sns.barplot(d.index,d.values,palette = 'coolwarm')
plt.xlabel('kills')
plt.ylabel('count')
plt.show()


# In[ ]:


#Bivariate analysis
#winPlacePerc vs walkDistance
#walkDistance shows strong positive correlation with winPlacePerc with a value of ,r = 0.8
#winPlacePerc increases with increse in walkDistance
sns.jointplot(x ='winPlacePerc',y='walkDistance',data = data1,height= 10,color='purple')
plt.show()


# In[ ]:


#winPlacePerc vs boosts
#we can see that number of boost items used is also positively correlated to winPlacePerc
ax = sns.jointplot(x ='winPlacePerc',y='boosts',data = data1,height=10,color='green')
plt.show()


# In[ ]:


#winPlacePerc vs weaponsAcquired
#Number of weapons picked up is positively correlated to winPlacePerc
ax = sns.jointplot(data1.winPlacePerc,data1.weaponsAcquired,height=10)
plt.show()


# In[ ]:


#winPlacePerc vs damageDealt
ax = sns.jointplot(data1.winPlacePerc,data1.damageDealt,height = 10,color ='violet')
plt.show()


# In[ ]:


#winPlacePerc vs heals
#Number of healing items used is also positively correlated to winPlacePerc
#winPlacePerc increases with increase in number of healing items used
ax = sns.jointplot(data1.winPlacePerc,data1.heals,height= 10,color='violet')
plt.show()


# In[ ]:


#Splitting Data into Training and Test set
X = data1.iloc[:,:24].values
Y = data1.iloc[:,-1].values
X_ = data2.iloc[:,1:].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30)


# In[ ]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 


# In[ ]:


#Training model using Regression
#First I have trained the model using multivariate regression
#But it gives accuracy around 83% on test set
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,Y_train)
lin.score(X_test,Y_test)


# In[ ]:


#Training model using RandomForestRegressor
#maximum depth of the tree = 10
#n_estimators parameter defines number of trees in the random forest.
#n_estimators = 100
#I have used mean squared error as the criterion to measure the quality of a split and we can also use mean absolute error.
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(max_depth = 10,criterion='mse',n_estimators=100)
reg.fit(X_train,Y_train)


# In[ ]:


#Predicting on test set
Y_pred = reg.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score
sc = r2_score(Y_test,Y_pred)
sc


# In[ ]:


import seaborn as sns 
from collections import Counter
from pprint import pprint
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
sample_submission = pd.read_csv('../input/sample_submission_V2.csv')


# In[ ]:


train = train.dropna()
train[train.killStreaks > 7].shape


# In[ ]:


#Lets add some more features
train['total_dist'] = train['swimDistance'] + train['walkDistance'] + train['rideDistance']

test['total_dist'] = test['swimDistance'] + test['walkDistance'] + test['rideDistance']

train['kills_with_assist'] = train['kills'] + train['assists']

test['kills_with_assist'] = test['kills'] + test['assists']


# In[ ]:


test['kills_with_assist'] = test['kills'] + test['assists']


# In[ ]:


train['headshot_over_kills'] = train['headshotKills'] / train['kills']
train['headshot_over_kills'].fillna(0, inplace=True)

test['headshot_over_kills'] = test['headshotKills'] / test['kills']
test['headshot_over_kills'].fillna(0, inplace=True)


# In[ ]:


train['headshot_over_kills'].value_counts().head(5)


# In[ ]:


train = train.drop(['Id','groupId','matchId'],axis=1)


# In[ ]:


matchtype = train.matchType.unique()


# In[ ]:


matchtype.__len__()


# In[ ]:


match_dict = {}
for i,each in enumerate(matchtype):
    match_dict[each] = i


# In[ ]:


match_dict


# In[ ]:


train.matchType = train.matchType.map(match_dict)


# In[ ]:


matchtype_test = test.matchType.unique()
match_dict_test = {}
for i,each in enumerate(matchtype_test):
    match_dict_test[each] = i
test.matchType = test.matchType.map(match_dict_test)


# In[ ]:


y = train['winPlacePerc']


# In[ ]:


X = train.drop(['winPlacePerc'],axis=1)


# In[ ]:


X.shape,y.shape


# In[ ]:


y[:2]


# In[ ]:


X[:2]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


#Lets Normalize the train data
sc_X = StandardScaler()
X_trainsc = sc_X.fit_transform(X_train)
X_testsc = sc_X.transform(X_test)


# In[ ]:


#Linear Regression
lr = LinearRegression()
lr.fit(X_trainsc, y_train)

y_pred = lr.predict(X_testsc)

y_pred[:10]


# In[ ]:


rmse = sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)


# In[ ]:


r2 = r2_score(y_test,y_pred)


# In[ ]:


print("RMSE = >",rmse)
print("MSE = >",mse)
print("R Squared = >",r2)


# In[ ]:


dt = pd.DataFrame()
dt['Actual'] = y_test


# In[ ]:


dt['Predicted'] = y_pred


# In[ ]:


dt['Difference'] = abs(y_test-y_pred)


# In[ ]:


dt.head(10)


# In[ ]:


#LightGBM
parameters = {
                'max_depth': 1,'min_data_in_leaf': 85,'feature_fraction': 0.80,'bagging_fraction':0.8,'boosting_type':'gbdt',
                'learning_rate': 0.1, 'num_leaves': 30,'subsample': 0.8,'lambda_l2': 4,'objective': 'regression_l2',
                'application':'regression','num_boost_round':5000,'zero_as_missing': True,
                'early_stopping_rounds':100,'metric': 'mae','seed': 2
             }


# In[ ]:


train_data = lgb.Dataset(X_trainsc, y_train, silent=False)
test_data = lgb.Dataset(X_testsc, y_test, silent=False)
model = lgb.train(parameters, train_set = train_data,verbose_eval=500, valid_sets=test_data)


# In[ ]:


test.columns


# In[ ]:


#Lets check the prediction with x_testsc 
pred_lgb_samp_sc = model.predict(X_testsc, num_iteration = model.best_iteration)
lgb_res= pd.DataFrame()
lgb_res['Actual'] = y_test
lgb_res['Predicted_sc'] = pred_lgb_samp_sc
lgb_res['Difference'] = abs(y_test-pred_lgb_samp_sc)
lgb_res.head(10)


# In[ ]:


test = test.drop(['Id','groupId','matchId'],axis=1)


# In[ ]:


# We'll normalize the test data aswell for better prediction

sc_test = StandardScaler()
test_sc = sc_test.fit_transform(test)


# In[ ]:


test.columns


# In[ ]:


# prediction
pred_lgb_sc = model.predict(test_sc, num_iteration = model.best_iteration)
pred_lgb_sc[:10]


# In[ ]:


# Replace the prediction which is greator than 1 by 1 and less than 0 by 0

pred_lgb_sc[pred_lgb_sc > 1] = 1
pred_lgb_sc[pred_lgb_sc < 0] = 0


# In[ ]:


pred_lgb_sc.__len__()


# In[ ]:




