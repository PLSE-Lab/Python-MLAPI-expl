#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/odi-match-winner/Train.csv')
test = pd.read_csv('/kaggle/input/odi-match-winner/Test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe(include='O')


# In[ ]:


plt.figure(figsize=(14,8))
plt.subplot(211)
sns.countplot(train['Team1'])
plt.show()
plt.figure(figsize=(14,8))
plt.subplot(212)
sns.countplot(train['Team2'])
plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x="Team1", y="Team2", hue='Team1_Venue', data=train)


# In[ ]:


plt.figure(figsize=(8,6))
sns.scatterplot(x="Team1", y="Team2", hue='Team2_Venue', data=train)


# ****all analysis is done for TEAM 1****

# **Match played by team1 in HOME and NEUTRAL venue**

# In[ ]:


train[train['Team1_Venue'] == 'Home'].groupby(['Team1'])[['Team1']].count().plot(kind='bar', color='g', label='Home')

train[train['Team1_Venue'] == 'Neutral'].groupby(['Team1_Venue','Team1'])[['Team1']].count().plot(kind='bar', color='r')


# ******count of matches played by team in inning as first or second******

# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(train['Team1'], hue = train['Team1_Innings'])


# In[ ]:


train[train['Team1_Venue']=='Home'].groupby(['MatchWinner'])[['MatchWinner']].count().plot(kind='bar', figsize=(10,5))


# In[ ]:


plt.figure(figsize=(14,6))
sns.countplot(train['MatchWinner'], hue=train['Team1_Venue'])


# In[ ]:


plt.figure(figsize=(14,6))
sns.countplot(train['MatchWinner'], hue=train['Team1_Innings'])


# In[ ]:


a = train.groupby(['MonthOfMatch'])[['MonthOfMatch']].count()
a = (a.index)
for i in a:
    plt.figure(figsize=(14,6))
    plt.tight_layout()
    ax = sns.countplot(x='MatchWinner', data = train[train['MonthOfMatch']==i] , label=i)
    ax.legend(loc='best')
    plt.show()
    


# In[ ]:


sns.countplot(train['HostCountry'])


# Done with Visualization 

# In[ ]:


# dum_df = pd.get_dummies(data["Item_Category"], prefix='Type_is_' )
# dum_df
# data = data.join(dum_df)


# In[ ]:


data = pd.concat([train, test])


# In[ ]:


data.shape[0]


# In[ ]:


dum = pd.DataFrame()


# In[ ]:


data['Team1'] = data['Team1'].astype('O')
data['Team2'] = data['Team2'].astype('O')


# In[ ]:


for i in data.columns:
    if data[i].dtype == "O":
        print(i)
        dum_df = pd.get_dummies(data[i],prefix=i)
        
        dum = pd.concat([dum,dum_df], axis=1)
       
        


# In[ ]:


dum.columns


# In[ ]:


remove = ['Team1',
'Team2',
'Team1_Venue',
'Team2_Venue',
'Team1_Innings',
'Team2_Innings',
'MonthOfMatch']


# In[ ]:


for i in remove:
    print(i)
    data.drop(columns=i, inplace=True)


# In[ ]:


data = pd.concat([data, dum], axis=1)


# In[ ]:


data


# In[ ]:





# In[ ]:


train_data = data.iloc[:train.shape[0]]


# In[ ]:


test_data = data.iloc[train.shape[0]:]
test_data.drop(columns=('MatchWinner'), inplace=True)


# In[ ]:


x = train_data.drop(columns=['MatchWinner'])
y = train_data['MatchWinner']


# In[ ]:


y.astype('O')


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.05, random_state=7)


# In[ ]:


from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_data = sc.transform(test_data)


# In[ ]:


model = XGBClassifier(
    learning_rate = 0.01,
    n_estimators = 50000,
    max_depth = 1,
    colsample_bytree = 0.8,
    seed = 100,
    eval_metric = 'mlogloss')

model.fit(X_train, y_train, eval_metric='mlogloss',
         eval_set=[(X_test, y_test)],
         early_stopping_rounds = 100,
         verbose=100)


# In[ ]:


model_xb = XGBClassifier(
    learning_rate = 0.01,
    n_estimators = 9883,
    max_depth = 1,
    colsample_bytree = 0.8,
    seed = 100)


# In[ ]:


model_xb.fit(X_train,y_train)


# In[ ]:


preds = model_xb.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,log_loss,f1_score
print(accuracy_score(preds, y_test))


# In[ ]:



# from sklearn.model_selection import cross_val_score
# score=cross_val_score(X=x,y=y,estimator=model_xb,scoring='neg_log_loss',cv=5)
# np.mean(score)


# In[ ]:


y_pred=model_xb.predict_proba(test_data)
submission=pd.DataFrame(y_pred)

submission.head()


# # LIST OF MODELS 

# In[ ]:



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import LabelPowerset


# In[ ]:


# rf_random.best_params_

