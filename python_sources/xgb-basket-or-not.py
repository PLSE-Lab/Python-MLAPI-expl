#!/usr/bin/env python
# coding: utf-8

# **Import libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Summarize data**

# In[ ]:


data=pd.read_csv('../input/data.csv')


# In[ ]:


data.info()


# In[ ]:


object_vars=[var for var in data if data[var].dtype=='object']
numerical_vars=[var for var in data if data[var].dtype=='float' or data[var].dtype=='int']
for var in object_vars:
    print(data[var].value_counts())


# We notice that variables team_id and team_name only have one value each, so they are useless.

# In[ ]:


#drop team id and team name columns(only one value - LAL)
data=data.drop(['team_id','team_name'],axis=1)
#set game date as datetime


# The matchup variable contains information on the opponent, as well as where the game was played. Since we already have a variable to represent opponent, we only extract the ''home or away'' information from this variable.

# In[ ]:


#get info from matchup: home or away?
data['home']=data['matchup'].apply(lambda x: 1 if 'vs' in x else 0)
data=data.drop('matchup',axis=1)


# Visualize the location variables. These seem to refer to the location on the court where Kobe shot from. Since the two groups of variables are correlated, only one will be kept.

# In[ ]:


#plt.subplot()
plt.figure(figsize=(5,5))
plt.scatter(x=data['loc_x'],y=data['loc_y'],alpha=0.02)

plt.figure(figsize=(5,5))
plt.scatter(x=data['lon'],y=data['lat'],alpha=.02)


# In[ ]:


#unnecessary columns, since they are correlated to loc_x and loc_y
data=data.drop(['lon','lat'],axis=1)


# Add the minutes variable to the seconds variable to create a new variable representing total remaining time in seconds. It seems like in the last 4 seconds of games, Kobe's rate of success is significantly lower. We create a variable as a flag for time lower than 4 seconds.

# In[ ]:


data['time_remaining_seconds']=data['minutes_remaining']*60+data['seconds_remaining']
data=data.drop(['minutes_remaining','seconds_remaining'],axis=1)


# In[ ]:


data['time_remaining_seconds']
data['last_3_seconds']=data.time_remaining_seconds.apply(lambda x: 1 if x<4 else 0)


# Shot_id is a unique identifier of each shot - useless in the model

# In[ ]:


#drop shot_id column (useless)
data=data.drop('shot_id',axis=1)


# Visualize the success rate for the different shot types. For the action_type variable, the 20 least common values are replaced with ''Other"

# In[ ]:


#visualize difference between shot types
fig,ax=plt.subplots()
sn.barplot(x='combined_shot_type',y='shot_made_flag',data=data)
#Replace the 20 least common action types with value 'Other'
rare_action_types=data['action_type'].value_counts().sort_values(ascending=True).index.values[:20]
data.loc[data['action_type'].isin(rare_action_types),'action_type']='Other'


# Check if the three variables contain independent information and cannot be inferred from each other.

# In[ ]:


#keep the three columns, they are not redundant
pd.DataFrame({'area':data.shot_zone_area,'basic':data.shot_zone_basic,'range':data.shot_zone_range}).head(10)


# The proportion of successful shots is similar in the two cases and does not seem to depend on the playoff variable. Therefore, this variable is dropped.

# In[ ]:


#drop playoffs - not relevant
sn.countplot('playoffs',hue='shot_made_flag',data=data)
data=data.drop('playoffs',1)


# Since the season variable represents the year, we only need to extract the month from the date variable. The weekday is unlikely to be relevant.

# In[ ]:


#get month from date
data['game_date']=pd.to_datetime(data['game_date'])
data['game_month']=data['game_date'].dt.month
data=data.drop('game_date',axis=1)


# These variables uniquely represent each game and each event. They are unlikely to be relevant to the success of a shot.

# In[ ]:


data=data.drop(['game_id','game_event_id'],axis=1)


# In[ ]:


data.info()


# The categorical variables are encoded as dummy variables, since tree-based algorithms do not accept string input.

# In[ ]:


#get dummies from categorical data
categorical_vars=['action_type','combined_shot_type','season','opponent','shot_type','period','shot_zone_basic','shot_zone_area','shot_zone_range','game_month']
for var in categorical_vars:
        data=pd.concat([data,pd.get_dummies(data[var],prefix=var)], 1)
        data=data.drop(var,1)


# **Setting target variable and train and test sets**

# In[ ]:


#separate train and test sets
train=data[pd.notnull(data['shot_made_flag'])]
test=data[pd.isnull(data['shot_made_flag'])]
y_train=train['shot_made_flag']
train=train.drop('shot_made_flag',1)
y_train=y_train.astype('int')
test=test.drop('shot_made_flag',1)


# In[ ]:


train.info()


# **Bivariate analysis**

# In[ ]:


#Correlation between numerical variables and shots made
sn.heatmap(data.corr())


# **Set log_loss as the scorer for the model**

# In[ ]:


#Evaluation with log loss
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
#log_scorer=make_scorer(log_loss,greater_is_better=False)
def log_scorer(estimator, X, y):
    pred_probs = estimator.predict_proba(X)[:, 1]
    return log_loss(y, pred_probs)


# Compare RandomForestClassifier and XGBClassifier models with default parameters. Use 5-fold cross-validation to obtain average test results.

# In[ ]:


#compare random forest and xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

models=[]
models.append(('RF',RandomForestClassifier()))
models.append(('XGB',XGBClassifier()))

results=[]

for name,model in models:
        cv=cross_val_score(model,train,y_train,scoring=log_scorer,cv=5)
        results.append((name,cv))
results


# **Hyperparameter tuning**

# Random Forest Classifier

# In[ ]:


from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier()
params={'n_estimators':[10,20,30,100,300],'max_depth':[5,10]}
grid=GridSearchCV(rf, param_grid=params, scoring=log_scorer, cv=5)
grid.fit(train, y_train)

best_max_depth=grid.best_params_['max_depth']
best_n_estimators=grid.best_params_['n_estimators']
print(best_max_depth,best_n_estimators)
pred=grid.predict_proba(train)[:,1]
print("Log loss for training set: ", log_loss(y_train,pred))
df=pd.DataFrame(grid.cv_results_)
target_y=grid.predict_proba(test)[:,1]


# XGB Classifier

# In[ ]:


params={'n_estimators': [300,400],
        'max_depth': [6,7,8],
        'learning_rate': [0.01,0.1,1],
        'subsample': [0.5,1],
        'colsample_bytree': [0.8,1],
        'seed': [0,1234]}
grid=GridSearchCV(XGBClassifier(warm_start=True), param_grid=params, cv=5, scoring=log_scorer)
grid.fit(train,y_train)
#print(grid.best_params_)
#print(grid.best_score_)
pred=grid.predict_proba(train)[:,1]
print("Log loss for training set: ", log_loss(y_train,pred))
grid.grid_scores_


# **Results**
# 
# random forest predict_proba train: 0.5626, test: 0.6120
# 
# best params xgboost: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 400, 'seed': 1234, 'subsample': 0.5}
# 
# xgb predict_proba train: 0.5686, test: 0.60063
# 

# **Fit the final model with the best parameters**

# In[ ]:


model=XGBClassifier(colsample_bytree= 0.8, learning_rate= 0.01, max_depth= 7, n_estimators= 400, seed= 1234, subsample= 0.5)
model.fit(train, y_train)
target_y = model.predict_proba(test)[:,1]


# **Write final predictions to submission file**

# In[ ]:


sub = pd.read_csv("../input/sample_submission.csv")
sub['shot_made_flag'] = target_y
sub.to_csv("submission.csv", index=False)

