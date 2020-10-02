#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/moneyball"))
print(os.listdir("../input/train"))

# Any results you write to the current directory are saved as output.


# Ok, first I will load data. For some reason the train.csv is no available on input folder. So I uploaded it.

# In[2]:


train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/moneyball/test.csv')


# In[3]:


print(train.head(2))
print(test.head(2))


# Ok, what is the meaning of the columns. I am from Argentina, I don't know anybody about baseball. So, this will be a little difficult. 

# In[4]:


print(train.columns)


# ## Dataset description
# I used the [MLB Glossay](http://m.mlb.com/glossary/)
# 
# **ID**:  interpreting the dataset I suppose that this is the ID player (?)
# 
# **W**: Win.  A pitcher receives a win when he is the pitcher of record when his team takes the lead for good 
# 
# **G**: Games played. A player is credited with having played a game if he appears in it at any point
# 
# **R**: Run. A player is awarded a run if he crosses the plate to score his team a run. When tallying runs scored, the way in which a player reached base is not considered. If a player reaches base by an error or a fielder's choice, as long as he comes around to score, he is still credited with a run. If a player enters the game as a pinch-runner and scores, he is also credited with a run.
# 
# **AB**: At-Bat. An official at-bat comes when a batter reaches base via a fielder's choice, hit or an error (not including catcher's interference) or when a batter is put out on a non-sacrifice. (Whereas a plate appearance refers to each completed turn batting, regardless of the result.)
# 
# **H**: Hit. A hit occurs when a batter strikes the baseball into fair territory and reaches base without doing so via an error or a fielder's choice. There are four types of hits in baseball: singles, doubles, triples and home runs. All four are counted equally when deciphering batting average. If a player is thrown out attempting to take an extra base (e.g., turning a single into a double), that still counts as a hit.
# 
# **1B**: Single. A single occurs when a batter hits the ball and reaches first base without the help of an intervening error or attempt to put out another baserunner
# 
# **2B**: Double. A batter is credited with a double when he hits the ball into play and reaches second base without the help of an intervening error or attempt to put out another baserunner.
# 
# **3B**: Triple. Often called "the most exciting play in baseball," a triple occurs when a batter hits the ball into play and reaches third base without the help of an intervening error or attempt to put out another baserunner.
# 
# **HR**: Home Run. Ok, I don't have doubt about this. But here the definition: A home run occurs when a batter hits a fair ball and scores on the play without being put out or without the benefit of an error.
# 
# **BB**: Walk: A walk occurs when a pitcher throws four pitches out of the strike zone, none of which are swung at by the hitter. After refraining from swinging at four pitches out of the zone, the batter is awarded first base.
# 
# **SO**: Strikeout. A strikeout occurs when a pitcher throws any combination of three swinging or looking strikes to a hitter. (A foul ball counts as a strike, but it cannot be the third and final strike of the at-bat. A foul tip, which is caught by the catcher, is considered a third strike.)
# 
# **SB**: Stolen Base. A stolen base occurs when a baserunner advances by taking a base to which he isn't entitled. This generally occurs when a pitcher is throwing a pitch, but it can also occur while the pitcher still has the ball or is attempting a pickoff, or as the catcher is throwing the ball back to the pitcher.
# 
# **CS**: Caught Stealing. A caught stealing occurs when a runner attempts to steal but is tagged out before reaching second base, third base or home plate. This typically happens after a pitch, when a catcher throws the ball to the fielder at the base before the runner reaches it. But it can also happen before a pitch, typically when a pitcher throws the ball to first base for a pickoff attempt but the batter has already left for second.
# 
# **HBP**: Hit-by-pitch. A hit-by-pitch occurs when a batter is struck by a pitched ball without swinging at it. He is awarded first base as a result. Strikes supersede hit-by-pitches, meaning if the umpire rules that the pitch was in the strike zone or that the batter swung, the HBP is nullified.
# 
# **BBHBP**: (?)
# 
# **SF**: Sacrifice Fly. A sacrifice fly occurs when a batter hits a fly-ball out to the outfield or foul territory that allows a runner to score. The batter is given credit for an RBI. (If the ball is dropped for an error but it is determined that the runner would have scored with a catch, then the batter is still credited with a sacrifice fly.)
# 
# **Outs**: Out (?). One of baseball's most basic principles, an out is recorded when a player at bat or a baserunner is retired by the team in the field. Outs are generally recorded via a strikeout, a groundout, a popout or a flyout, but MLB's official rulebook chronicles other ways -- including interfering with a fielder -- by which an offensive player can be put out.
# 
# **Outsinplay**: (?)
# 
# **RA**: (?)
# 
# **BA**:  (?)
# 
# **OBA**: (?)
# 
# **SLG**: Slugging Percentage. Slugging percentage represents the total number of bases a player records per at-bat. Unlike on-base percentage, slugging percentage deals only with hits and does not include walks and hit-by-pitches in its equation.
# 
#   **OPS**: On-base Plus Slugging. 
# 

# # Exploring data
# 

# In[5]:


train.describe()


# We can see that the min and max and std say us that there are variability on the data

# In[6]:


print(train.isnull().sum())
print(train.info())


# Wow, we don't have missing value. What about on test fiel?

# In[7]:


print(test.isnull().sum())
print(test.info())


# Ok, that is good. 

# Ok, let's see some data visualisation to try learn about some data.

# In[8]:


plt.figure(figsize=(14,12))
foo = sns.heatmap(train.drop('ID',axis=1).corr(), vmax=0.6, square=True, annot=True)


# Now, we continue examine the correlations data. 
# 
# 

# In[9]:


fig, (axarr1, axarr2, axarr3) = plt.subplots(3, 3, figsize=(12, 12))
# sns.kdeplot(train.W, ax=axarr1[0])
sns.kdeplot(train.W, train.R, ax=axarr1[0])
axarr1[0].set_title("W vs R")
sns.kdeplot(train.W, train.RA, ax=axarr1[1]) 
axarr1[1].set_title("W vs RA")
sns.kdeplot(train.W, train.H, ax=axarr1[2])
axarr1[2].set_title("W vs H")
sns.kdeplot(train.W, train.HR, ax=axarr2[0])
axarr2[0].set_title("W vs HR")
sns.kdeplot(train.W, train['1B'], ax=axarr2[1])
axarr2[1].set_title("W vs 1B")
sns.kdeplot(train.W, train['2B'], ax=axarr2[2])
axarr2[2].set_title("W vs 2B")
sns.kdeplot(train.W, train['3B'], ax=axarr3[0])
axarr3[0].set_title("W vs 3B")
sns.kdeplot(train.W, train.BB, ax=axarr3[1])
axarr3[1].set_title("W vs BB")
sns.kdeplot(train.W, train.SO, ax=axarr3[2])
axarr3[2].set_title("W vs SO")


# In other kernel I will work with engineering data
# 
# # Preparing for modeling
# 

# In[10]:


train.columns


# In[11]:


cols = ['G', 'R', 'AB', 'H', 'BB','BBHBP', 'Outs', 'RA', 'BA', 'OPS', '3B', 'SO', 'CS']
ttrain, ttest = train_test_split(train, test_size = 0.2)
train_x = ttrain[cols]
#rain_x = ttrain
train_target = ttrain.W

test_x = ttest[cols]
#est_x = ttest
test_target = ttest.W


# We will try to automatize feature selection

# In[12]:


# feature extraction
train_cols = SelectKBest(score_func=chi2, k=15)
fit = train_cols.fit(ttrain.drop(['W', 'ID'], axis=1), ttrain.W)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(ttrain.drop(['W', 'ID'], axis=1))
# Summarize selected features
print(features[0:5, :])
print(fit.get_support())


# In[13]:


dummy = ttrain.drop(['W', 'ID'], axis=1)
dummy.reset_index(drop=True, inplace=True)
dummy.iloc[:, fit.get_support()]
dummy_columns = ttrain.drop(['W', 'ID'], axis=1).columns
#print(dummy[fit.get_support()])


# In[14]:


columns_selectkBest = dummy_columns[fit.get_support()]
train_x_selectKBest = ttrain[columns_selectkBest]
test_x_selectKBest = ttest[columns_selectkBest]


# In[15]:


regressor = linear_model.LinearRegression()
rfe = RFE(regressor, 20)
fit = rfe.fit(ttrain.drop(['ID', 'W'], axis=1), ttrain.W)
print("Num Features: {}".format(fit.n_features_)) 
print("Selected Features: {}".format(fit.support_))
print("Feature Ranking: {}".format(fit.ranking_))

columns_rfe = dummy_columns[fit.get_support()]
train_x_rfe = ttrain[columns_rfe]
test_x_rfe = ttest[columns_rfe]
print(train_x_rfe.columns)
print(test_x_rfe.columns)


# # Modeling
# First, I will start with a simple Linear models.
# 

# In[16]:


regressor = linear_model.LinearRegression()
regressor.fit(train_x, train_target)
pred =  regressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
print('Coefficients: \n', regressor.coef_)


# In[17]:


regressor_selectKBest = linear_model.LinearRegression()
regressor_selectKBest.fit(train_x_selectKBest, train_target)
pred = regressor_selectKBest.predict(test_x_selectKBest)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
print('Coefficients: \n', regressor.coef_)


# In[18]:


regressor_rfe = linear_model.LinearRegression()
regressor_rfe.fit(train_x_rfe, train_target)
pred = regressor_rfe.predict(test_x_rfe)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[19]:


print(len(ttrain.columns))


# In[22]:


regressor = linear_model.LinearRegression()
r2_score_dict = dict()
for i in range(2, 24):
    rfe = RFE(regressor, i)
    fit = rfe.fit(ttrain.drop(['ID', 'W'], axis=1), ttrain.W)
    columns_rfe = dummy_columns[fit.get_support()]
    train_x_rfe = ttrain[columns_rfe]
    test_x_rfe = ttest[columns_rfe]
    regressor_rfe = linear_model.LinearRegression()
    regressor_rfe.fit(train_x_rfe, train_target)
    pred = regressor_rfe.predict(test_x_rfe)
    r2_score_dict[i] = r2_score(test_target, pred)
# print("Num Features: {}".format(fit.n_features_)) 
# print("Selected Features: {}".format(fit.support_))
# print("Feature Ranking: {}".format(fit.ranking_))
# print(train_x_rfe.columns)
# print(test_x_rfe.columns)
print(max(r2_score_dict, key=r2_score_dict.get))


# In[25]:


regressor = linear_model.LinearRegression()
rfe = RFE(regressor, 22)
fit = rfe.fit(ttrain.drop(['ID', 'W'], axis=1), ttrain.W)
columns_rfe = dummy_columns[fit.get_support()]
train_x_rfe = ttrain[columns_rfe]
test_x_rfe = ttest[columns_rfe]
regressor_rfe2 = linear_model.LinearRegression()
regressor_rfe2.fit(train_x_rfe, train_target)
pred = regressor_rfe2.predict(test_x_rfe)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit(train_x,  train_target)
print(reg.alpha_)


# In[ ]:


ridgeRegressor = linear_model.Ridge(alpha=0.1)
ridgeRegressor.fit(train_x, train_target)
pred = ridgeRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


lassoRegressor = linear_model.Lasso(alpha=0.1)
lassoRegressor.fit(train_x, train_target)
pred = lassoRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


elasticRegressor = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
elasticRegressor.fit(train_x, train_target)
pred = elasticRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


lassoLarsRegressor = linear_model.LassoLars(alpha=0.1)
lassoLarsRegressor.fit(train_x, train_target)
pred = lassoLarsRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


bayesianRidge = linear_model.BayesianRidge()
bayesianRidge.fit(train_x, train_target)
pred = bayesianRidge.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


ardRegression = linear_model.ARDRegression(compute_score=True)
ardRegression.fit(train_x, train_target)
pred = ardRegression.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


huberRegressor = linear_model.HuberRegressor()
huberRegressor.fit(train_x, train_target)
pred = huberRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


import decimal
def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)


# In[ ]:


epsilon = drange(1, 10, 0.1)
dict_r2_score = dict()
for e in epsilon:
    huber = linear_model.HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,
                           epsilon=e)
    huber.fit(train_x, train_target)
    p = huber.predict(test_x)
    dict_r2_score[e] = r2_score(test_target, p)


# In[ ]:


max_id = max(dict_r2_score, key=dict_r2_score.get)
print(max_id, dict_r2_score[max_id])


# In[ ]:


huberRegressor = linear_model.HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, 
                                            epsilon=5.8)
huberRegressor.fit(train_x, train_target)
pred = huberRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


randomForest = ensemble.RandomForestRegressor(n_estimators=100)
rf_params = {'max_depth': range(5, 8),
             'max_features': range(5, 10)}


# In[ ]:


cv = GridSearchCV(cv=5, param_grid=rf_params, estimator=randomForest, n_jobs=-1, scoring='neg_mean_absolute_error')


# In[ ]:


cv.fit(train_x, train_target)


# In[ ]:


print(-cv.best_score_)
print(cv.best_params_)


# In[ ]:


randomForest = ensemble.RandomForestRegressor(n_estimators=100, max_depth=7, max_features=9, n_jobs=-1)
randomForest.fit(train_x, train_target)
pred = randomForest.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[ ]:


randomForest2 = ensemble.RandomForestRegressor(n_estimators=100)
rf_params = {'max_depth': range(5, 15),
             'max_features': range(5, 23)}
cv = GridSearchCV(cv=5, param_grid=rf_params, estimator=randomForest, n_jobs=-1, scoring='neg_mean_absolute_error')
cv.fit(train_x, train_target)
print(-cv.best_score_)
print(cv.best_params_)


# In[ ]:


randomForest2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=14, max_features=22, n_jobs=-1)
randomForest2.fit(train_x, train_target)
pred = randomForest2.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))


# In[27]:


# pred_submit = ridgeRegressor.predict(test)
# pred_submit = huberRegressor.predict(test)
pred_submit = regressor_rfe2.predict(test[columns_rfe])
submit = pd.DataFrame(pred_submit, index=test['ID'], columns=['W'])
submit = submit.round()
submit.W = submit.W.astype('int64')
submit.to_csv('regressor_rfe2.csv', index_label='ID')


# In[ ]:




