#!/usr/bin/env python
# coding: utf-8

# 2018-04-05

# # Introduction

# * Load Data
#     * Data summary
# * Target
#     * outlier
# * Feature engineering
#     * Datetime
#     * Categorical&Numeric
# * Visualization
#     * Categorical
#     * Numerical
#     * All Feature
# * Modeling

# # Load Data

# ##  Data Summary

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
from scipy import stats
pd.options.mode.chained_assignment = None
from scipy.stats import norm, skew
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# Fortunately, the Missing value of the entire data does not seem to exist.

# In[ ]:


print('Train Dataset Shape : {0}'.format(train.shape))
print('Test Dataset Shape : {0}'.format(test.shape))


# In[ ]:


train.dtypes


# # Target

# ## Outlier

# In[ ]:


sns.boxplot(train['count'])


# You can judge there is an extlier by count.
# It is good to remove outlier because it has a detrimental effect on the model. So, first, you have to remove it before you analyze it.

# In[ ]:


train = train[np.abs(train["count"]-train["count"].mean())<=(3*train["count"].std())] 


# In[ ]:


fig,ax = plt.subplots(2,1,figsize = (10,10))
sns.distplot(train['count'],ax=ax[0])
stats.probplot(train["count"], dist='norm', fit=True, plot=ax[1])
print('Skewness : {0}'.format(train['count'].skew()))
print('Kurt : {0}'.format(train['count'].kurt()))


# The graph above shows the distribution chart of Count. As a graph before the log is processed on the left, you can see that the concentration is very high between 0 and 200. This means that it follows the normal distribution, as shown in the graph on the right, because it has a shifted power distribution.

# In[ ]:


fig,ax = plt.subplots(2,1,figsize = (10,10))
#logcount = np.log1p(train['count']).kurt()
#rootcount = np.sqrt(train['count']).kurt()
#cubiccount = np.power(train['count'],2).kurt()
#minVal = min([logcount, rootcount, cubiccount])
#if logcount == minVal:
best = 'log'
train['count_log'] = np.log1p(train['count'])
sns.distplot(train['count_log'],ax=ax[0])
stats.probplot(train["count_log"], dist='norm', fit=True, plot=ax[1])
#elif rootcount == minVal:
    #best = 'root'
    #train['count_root'] = np.sqrt(train['count'])
    #sns.distplot(train['count_root'],ax=ax[0])
    #stats.probplot(train["count_root"], dist='norm', fit=True, plot=ax[1])
#elif cubiccount == minVal:
    #best = 'cubic'
    #train['count_cubic'] = np.power(train['count'],2)
    #sns.distplot(train['count_cubic'],ax=ax[0])
    #stats.probplot(train["count_cubic"], dist='norm', fit=True, plot=ax[1])
#print('For count, the Best TF is ' + best)


# # Feature engineering 

# Well, before we make a quick analysis, let's make some new features.
# At first glance at the data, the datetime is likely to be used in some significant way.
# I have to check the number of bicycles I borrow regularly.
# Let's touch the datetime first.

# * > Season - (spring, 2 : summer, 3 : fall, 4 : winter)
# * > Holiday considerations
# * > Workingday
# * > Weather - (1 : Sunny, 2 : Mist, 3 : light snow, rain, 4 : heavy snow, rain)
# * > Temp-temperature
# * > Atemp - weather (impression index)
# * > Windspeed
# * > Casual - the number of times an unregistered user lease has started
# * > Register - starts with the number of registered user rentals
# * > Count - Total lease count (dependent variables)

# ## Datetime

# In[ ]:


train['date']  = train.datetime.apply(lambda x: x.split()[0])
train['hour'] = train.datetime.apply(lambda x: x.split()[1].split(':')[0])
train['weekday'] = train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
train['month'] = train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)
train = train.drop('datetime',axis=1)


# In[ ]:


train.shape


# # Categorical & Numeric

# In[ ]:


train.dtypes


# In[ ]:


categorical = ['date','weekday','month','hour','season','holiday','workingday','weather']
numeric = ["temp","atemp","casual","registered","humidity","windspeed","count","count_log"]


# In[ ]:


for idx in categorical:
    train[idx].astype('category')


# # Visualization

# # Categorical 

# In[ ]:


fig,axes = plt.subplots(ncols=2 ,nrows=2)
fig.set_size_inches(15,10)
sns.boxplot(data=train,x='season',y='count',ax=axes[0][0])
sns.boxplot(data=train,x='holiday',y='count',ax=axes[0][1])
sns.boxplot(data=train,x='workingday',y='count',ax=axes[1][0])
sns.boxplot(data=train,x='weather',y='count',ax=axes[1][1])

fig1,axes1 = plt.subplots()
fig1.set_size_inches(15,10)
sns.boxplot(data=train,x='hour',y='count')


# You can see a few things from the graph above.
# * > Season can see that spring has a lower count than summer, fall, and winter.
# * > You can see there are quite a few outliers for Count. 3. When looking at Workingday, outlier is higher when working than when not working.(Use more when working.)
# * > The hour shows the most distribution in the 08 AM and 17 PM.

# # Numercial 

# In[ ]:


plt.subplots(figsize=(15,8))
sns.heatmap(train[numeric].corr(),annot=True)


# In general, you can see that most variables have a lot to do with Count. So let's take a look at the relationship between the different variables.

# In[ ]:


corr = train[numeric].drop('count', axis=1).corr()
corr =corr.drop('count_log', axis=1).corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# Temp and atemp appear to be too closely associated with each other. 
# If this occurs, then remove the atemp later because creating a model that predicts dependent variables may prevent the correct results.

# ## All Feature

# In[ ]:


### count,month
plt.figure(figsize=(15,8))
monthagg = pd.DataFrame(train.groupby('month')['count'].mean()).reset_index()
sns.barplot(data=monthagg, x='month',y='count').set(title = 'Month Vs Count')


# In[ ]:


### count,season,hour
plt.figure(figsize=(15,8))
houragg = pd.DataFrame(train.groupby(['hour','season'])['count'].mean()).reset_index()
sns.pointplot(data=houragg,x=houragg['hour'],y=houragg['count'],hue=houragg['season']).set(title='Hour,Season Vs Count')


# As we saw above, the usage in spring is significantly lower than in summer, fall, and winter, but it is most commonly used between 8 and 9 a.m. during rush hour and between 17 and 19 during rush hour.

# In[ ]:


### count,hour,weekday
plt.figure(figsize=(15,8))
hourweekagg = pd.DataFrame(train.groupby(['hour','weekday'])['count'].mean()).reset_index()
sns.pointplot(data=hourweekagg,x=hourweekagg['hour'],y=hourweekagg['count'],hue=hourweekagg['weekday']).set(title='Hour,Week Vs Count')


# You can find out interesting facts. On the weekends, they are relatively used in the afternoon, but from Monday to Friday, they have the most hours to leave work.
# The results came out as we had expected.

# # Modelling

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


target = train['count']
target_log=train['count_log']
train = train.drop('count_log',axis=1)
train = train.drop('count',axis=1)
train = train.drop('atemp',axis=1)
train = train.drop('date',axis=1)
train = train.drop('casual',axis=1)
train = train.drop('registered',axis=1)
m_dum = pd.get_dummies(train['month'],prefix='m')
ho_dum = pd.get_dummies(train['hour'],prefix='ho')
s_dum = pd.get_dummies(train['season'],prefix='s')
we_dum = pd.get_dummies(train['weather'],prefix='we')
train = pd.concat([train,s_dum,we_dum,m_dum,ho_dum],axis=1)

testid = test['datetime']
test['date']  = test.datetime.apply(lambda x: x.split()[0])
test['hour'] = test.datetime.apply(lambda x: x.split()[1].split(':')[0])
test['weekday'] = test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
test['month'] = test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)
test = test.drop('datetime',axis=1)
test = test.drop('atemp',axis=1)
test = test.drop('date',axis=1)
s_dum = pd.get_dummies(test['season'],prefix='s')
we_dum = pd.get_dummies(test['weather'],prefix='we')
m_dum = pd.get_dummies(test['month'],prefix='m')
ho_dum = pd.get_dummies(test['hour'],prefix='ho')
test= pd.concat([test,s_dum,we_dum,m_dum,ho_dum],axis=1)


# In[ ]:


train.shape


# In[ ]:


test.shape


# # GBR

# In[ ]:


gbr = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4).fit(train.values, target_log)


# In[ ]:


def loss_func(truth, prediction):
    y = np.expm1(truth)
    y_ = np.expm1(prediction)
    log1 = np.array([np.log(x + 1) for x in truth])
    log2 = np.array([np.log(x + 1) for x in prediction])
    return np.sqrt(np.mean((log1 - log2)**2))


# In[ ]:


#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import make_scorer
#param_grid = {
#    'learning_rate': [0.1, 0.01, 0.001],
#    'n_estimators': [100, 1000, 1500, 2000, 4000],
#    'max_depth': [1, 2, 3, 4, 5]
#}
#scorer = make_scorer(loss_func, greater_is_better=False)
#model = GradientBoostingRegressor(random_state=42)
#result = GridSearchCV(model, param_grid, cv=4, scoring=scorer, n_jobs=3).fit(train.values, target_log)
#print('\tParams:', result.best_params_)
#print('\tScore:', result.best_score_)


# In[ ]:


##	Params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1500}
#	Score: -0.12669018059776296


# In[ ]:


model_gbr = GradientBoostingRegressor(n_estimators=1500,max_depth=5,learning_rate=0.01).fit(train.values,target_log)


# In[ ]:


prediction = model_gbr.predict(test.values)
prediction = np.expm1(prediction)


# In[ ]:


output = pd.DataFrame()
output['datetime'] = testid
output['count'] = prediction
output.to_csv('output.csv',index=False)


# In[ ]:




