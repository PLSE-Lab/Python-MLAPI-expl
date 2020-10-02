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


# # PUBG DATA SET
# <img src="https://saudigamer.com/wp-content/uploads/2018/09/123.jpg" width="10000px">
# 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
print(os.listdir("../input"))


# # Import the Train Data Set

# In[ ]:


train_complete=pd.read_csv("../input/train_V2.csv")
train_complete.head()


# In[ ]:


train=train_complete.sample(100000,random_state =1)
train.head()


# **Removing the columns with unique id's**

# In[ ]:


train=train.drop(['Id','groupId','matchId'],axis=1)
train.head()


# In[ ]:


train.info()


# There are no null values and no non numeric data columns . We can proceed with the visuals

# In[ ]:


dftrain=train.copy()


# In[ ]:


corr=dftrain.corr()
corr


# In[ ]:



plt.figure(figsize=(15,10))
sns.heatmap(corr,annot=True)


# In[ ]:


plt.figure(figsize=(15,10))
corr1=corr.abs()>0.5
sns.heatmap(corr1,annot=True)


# ## Now lets the corelation of the variables with the target variable

# In[ ]:


plt.title('Correlation B/w Winning % and other Independent Variable')
dftrain.corr()['winPlacePerc'].sort_values(ascending=False).plot(kind='bar',figsize=(10,8))


# ### Here we can see that walk distance,boosts,weapons Acquired, damage dealt,heals, kills, long kills,kills streaks, ride distance show good corelation with the target variables

# ### Let's look at the top ten features which are corelated with the target variables.

# In[ ]:


k = 10 #number of variables for heatmap
f,ax = plt.subplots(figsize=(11, 11))
cols = dftrain.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index
cm = np.corrcoef(dftrain[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# ### Lets do a pair plot analysis for the top six features

# In[ ]:


sns.set()
cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']
sns.pairplot(dftrain[cols], size = 2.5)
plt.show()


# ## Now lets deal with the multicoliearilty and remove the less important variables
# - We will calculate the ** variance inflation factor**  for the complete data set.
# - Feature selection by step wise method and arrive at final features which gives the best accuracy
# - VIF = 1/(1-r2)
# - There is thumb rule that the variables which are more than 10 are highly corelated and hence we can remove then

# In[ ]:


train.info()


# **Here we need to convert the non numeric columns to numeric by the creating dummy variables**

# In[ ]:


train_complete=pd.get_dummies(train)
train_complete.head()


# **Variation Inflation Factor**

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
x_features=list(train_complete)
x_features.remove('winPlacePerc')
data_mat = train_complete[x_features].as_matrix()                                                                                                              
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif     
vif_factors.sort_values(by=['vif'],ascending=False)[0:10]


# We will try removing the variables and check the Variance Inflation Factor

# In[ ]:


x_features=list(train_complete)
x_features.remove('winPlacePerc')
x_features.remove('maxPlace')
x_features.remove('numGroups')
x_features.remove('winPoints')
x_features.remove('rankPoints')
x_features.remove('matchType_squad-fpp')
x_features.remove('matchDuration')
data_mat = train_complete[x_features].as_matrix()                                                                                                              
vif = [ variance_inflation_factor( data_mat,i) for i in range(data_mat.shape[1]) ]
vif_factors = pd.DataFrame()
vif_factors['column'] = list(x_features)
vif_factors['vif'] = vif     
vif_factors.sort_values(by=['vif'],ascending=False)[0:10]


# # Modelling the Data
# - Train Test Split
# - Model Comparision
# - Hyper Parameter Tuning
# - Building the Final Model

# #### Train Test Split

# In[ ]:


x=train_complete[x_features]
y=train_complete[['winPlacePerc']]
# Train Test Split
validation_size = 0.30
seed = 1
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)


# #### Model Comparision

# # Spot-Check Algorithms
# models = []
# models.append(('LR', LinearRegression()))
# models.append(('CART', DecisionTreeRegressor()))
# models.append(('GB', GradientBoostingRegressor()))# we can set the hyper parameters if needed

# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = KFold(n_splits=10, random_state=seed)
#     cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2') 
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, np.sqrt(cv_results.mean()), np.sqrt(cv_results.std()))
#     print(msg)

# #### Hyper Parameter Tuning

# # Gradient Boost Algorithm Hyper Parameter Tuning
# n_estimators = np.arange(50,110,10)
# learning_rate = np.arange(0.1,1.1,0.1)
# param_grid = dict(n_estimators=n_estimators,learning_rate=learning_rate)
# model = GradientBoostingRegressor()
# kfold = KFold(n_splits=5, random_state=seed)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5)
# grid_result = grid.fit(x_train, y_train)

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# #### Final Model

# In[ ]:


model_final=GradientBoostingRegressor(n_estimators=100,learning_rate=0.5)
model_final.fit(x_train,y_train)
print(model_final.score(x_train,y_train))
print(model_final.score(x_validation,y_validation))


# ## Importing Test Data

# In[ ]:


test=pd.read_csv('../input/test_V2.csv')
dftest=test.drop(['Id','groupId','matchId'],axis=1)
dftest.head()


# In[ ]:


dftest.info()


# In[ ]:


test_complete=pd.get_dummies(dftest)
x_test=test_complete[x_features]


# # Sample Submission

# In[ ]:


pred=model_final.predict(x_test)
pred[1:5]


# In[ ]:


pred_df=pd.DataFrame(pred,test['Id'],columns=['winPlacePerc'])
pred_df.head()


# In[ ]:


pred_df.to_csv('sample_submission.csv')

