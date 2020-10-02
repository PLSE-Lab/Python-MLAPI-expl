#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
from matplotlib.pylab import rcParams
from sklearn.metrics import accuracy_score
rcParams['figure.figsize'] = 12, 4
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load train and Test set
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[ ]:


#train.drop("soldierId", axis = 1, inplace = True)
#test.drop("soldierId", axis = 1, inplace = True)
train.drop("shipId", axis = 1, inplace = True)
test.drop("shipId", axis = 1, inplace = True)
train.drop("attackId", axis = 1, inplace = True)
test.drop("attackId", axis = 1, inplace = True)

#train.drop("castleTowerDestroys", axis = 1, inplace = True)
#test.drop("castleTowerDestroys", axis = 1, inplace = True)

#train.drop("horseRideKills", axis = 1, inplace = True)
#test.drop("horseRideKills", axis = 1, inplace = True)


# In[ ]:


#Correlation data picking first 10
k = 10 
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
cols = corrmat.nlargest(k, 'bestSoldierPerc')['bestSoldierPerc'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


#Extract train
train_input = train.drop('bestSoldierPerc', axis=1)
train_input = train_input.drop(train.columns[0], axis=1)
train_output = train['bestSoldierPerc']

# Fill Nulls
train_input.isnull().values.any()
train_input = train_input.fillna(train_input.mean())

#Scaling
scaler = StandardScaler()
scaler.fit(train_input)


# In[ ]:


train_input.head()


# In[ ]:


#Splitting and testing the moodel
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_input, train_output, test_size = 0.1, random_state=0)


# In[ ]:


import xgboost as xgb
xgb_model = xgb.XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.2,
 colsample_bytree=0.2,
 reg_alpha=0.005,
 objective= 'multi:softprobc',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_val)
acc_xgbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Accuracy with xgboost: ", acc_xgbk)
print()


# In[ ]:


gbk = GradientBoostingClassifier(learning_rate=0.005, n_estimators=2500,max_depth=8, min_samples_split=600, min_samples_leaf=50, subsample=0.95, random_state=0, max_features=4,
warm_start=True)
gbk.fit(x_train, y_train)
print("Learning rate: ", learning_rate)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Accuracy with gradientBoosting: ", acc_gbk)
print()


# In[ ]:


#Extract test
test_input = test.drop(train.columns[0], axis=1)

# Fill Nulls
test_input.isnull().values.any()
test_input = test_input.fillna(test_input.mean())

#Scaling
scaler = StandardScaler()
scaler.fit(test_input)

#Predict the output
predict = xgb_model.predict(test_input)
submission = pd.DataFrame({'soldierId': test_input['soldierId'],
                           'bestSoldierPerc' : predict})
submission.to_csv('Eval3submission79_29.csv',index=False)


# In[ ]:




