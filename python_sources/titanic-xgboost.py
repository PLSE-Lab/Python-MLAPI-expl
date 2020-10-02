#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import plotly
import scipy
import matplotlib
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import plotly.offline as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'notebook')
matplotlib.style.use('ggplot')


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv('../input/test.csv')
train_data.isnull().sum()
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_imputed= train_data


# In[ ]:


train_imputed.describe()
train_imputed.head()
train_imputed = train_imputed.drop(["Cabin","PassengerId","Name","Ticket"], axis =1)
mode_embarked = train_imputed['Embarked'].mode()[0]
train_imputed['Embarked'].fillna(mode_embarked,inplace=True)

dummy_embar = pd.get_dummies(train_imputed['Embarked'])
train_imputed = pd.concat([train_imputed, dummy_embar], axis= 1 )
dummy_pclass = pd.get_dummies(train_imputed['Pclass'])
train_imputed = pd.concat([train_imputed, dummy_pclass], axis= 1 )

train_imputed = train_imputed.drop("Embarked", axis =1)
train_imputed = train_imputed.drop("Pclass", axis =1)
train_imputed['Sex'] = train_imputed['Sex'].map({'male':1, 'female':0})
targets = train_imputed.pop('Survived')


# In[ ]:


test_data = test_data.drop(["Cabin","Name","Ticket"], axis =1)
passenger_ids = test_data.pop("PassengerId")
#print(test_data.isnull().sum())
#test_data['Embarked'].fillna(mode_embarked,inplace=True)
dummy_embar = pd.get_dummies(test_data['Embarked'])
test_data = pd.concat([test_data, dummy_embar], axis= 1 )
dummy_pclass = pd.get_dummies(test_data['Pclass'])
test_data = pd.concat([test_data, dummy_pclass], axis= 1 )
test_data = test_data.drop("Embarked", axis =1)
test_data = test_data.drop("Pclass", axis =1)
test_data['Sex'] = test_data['Sex'].map({'male':1, 'female':0})
testdmat = xgb.DMatrix(test_data)
print(test_data.head(4))


# In[ ]:


our_params = {'eta': 0.05, 'seed':0, 'subsample': 1, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':5, 'min_child_weight':5} 
xgdmat = xgb.DMatrix(train_imputed, targets)
final_gb = xgb.train(our_params, xgdmat, num_boost_round = 70)
xgb.plot_importance(final_gb)


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = final_gb.predict(testdmat) 
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
submission_df = pd.DataFrame({"PassengerID":passenger_ids,"Survived":y_pred})
submission_df.to_csv("submission.csv", index = False)

