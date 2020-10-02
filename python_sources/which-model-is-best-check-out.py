#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import RFE
import gc

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import catboost as cat
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import accuracy_score,roc_auc_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv', low_memory=True)
test = pd.read_csv('../input/test.csv', low_memory=True)
print(train.shape, test.shape)


# In[ ]:


train.head()


# In[ ]:


train.target.value_counts().plot(kind="barh")


# In[ ]:


y = train.pop('target')
train.pop('id')
X = train
test.pop('id')


# In[ ]:


X.shape, y.shape


# In[ ]:


Classifier = [LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(),RandomForestClassifier(), 
               AdaBoostClassifier(), GradientBoostingClassifier(),ExtraTreesClassifier(n_estimators=200),
              LinearSVC(),SVC(), NuSVC(),xgb.XGBClassifier(), lgb.LGBMClassifier(),cat.CatBoostClassifier()]
Classifier_name = ["Logistic Regression","Decision Tree","KNN","Random Forest",
                   "Adaboost","Gradient Boosting",
                   "Extra Tree",
                   "Linear SVC","SVC","NuSVC","XGB","LGB","CATBOOST"]


# In[ ]:


test.shape


# In[ ]:


from sklearn import *

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


Score = []
for i,j in zip(Classifier,Classifier_name):
#     print(j +" Classifier Training...")
    reg = i
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    print('ROC of '+j+':', metrics.roc_auc_score(y_test, pred))
    Score.append(metrics.roc_auc_score(y_test, pred))


# In[ ]:


graphdf = pd.DataFrame()
graphdf["Classifier_name"] = Classifier_name
graphdf["Score"] = Score


# In[ ]:


graphdf = graphdf.sort_values(by="Score", ascending=False)
plt.figure(figsize=(20,8))
sns.barplot(x=graphdf.Classifier_name,y = graphdf.Score,orient='v')
plt.legend()
plt.grid()
plt.xlabel("Models")
plt.ylabel("ROC AUC")


# In[ ]:


reg = GradientBoostingClassifier()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print('ROC of :', metrics.roc_auc_score(y_test, pred))


# In[ ]:


submission = pd.read_csv("../input/sample_submission.csv")
submission["target"] = reg.predict(test)
submission.to_csv('submission.csv', index=False)

