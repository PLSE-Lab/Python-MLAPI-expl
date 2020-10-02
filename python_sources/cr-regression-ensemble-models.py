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


placement = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

placement


# # 1. Building Linear Regression Model, with ssc_p, hsc_p, and degree_p as independent variables and mba_p as dependent variable.

# In[ ]:


task_1_data = placement.drop(columns=['sl_no','gender','ssc_b','hsc_b','hsc_s','degree_t','workex','etest_p','specialisation','status','salary'])

task_1_data


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

#Plotting ssc_p against mba_p
plt.subplot(2,2,1)
sns.regplot(x = task_1_data.ssc_p, y = task_1_data.mba_p)

#Plotting hsc_p against mba_p
plt.subplot(2,2,2)
sns.regplot(x = task_1_data.hsc_p, y = task_1_data.mba_p)

#Plotting degree_p against mba_p
plt.subplot(2,2,3)
sns.regplot(x = task_1_data.degree_p, y = task_1_data.mba_p)


# In[ ]:


#Standardizing data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
task_1_data[['hsc_p']] = scaler.fit_transform(task_1_data[['hsc_p']])

task_1_data[['ssc_p']] = scaler.fit_transform(task_1_data[['ssc_p']])

task_1_data[['degree_p']] = scaler.fit_transform(task_1_data[['degree_p']])

task_1_data[['mba_p']] = scaler.fit_transform(task_1_data[['mba_p']])


# In[ ]:


#Plotting scaled data. Note the difference along the X-axis and Y-axis units.

#Plotting ssc_p against mba_p
sns.regplot(x = task_1_data.ssc_p, y = task_1_data.mba_p)

#Plotting hsc_p against mba_p
sns.regplot(x = task_1_data.hsc_p, y = task_1_data.mba_p)

#Plotting degree_p against mba_p
sns.regplot(x = task_1_data.degree_p, y = task_1_data.mba_p)


# In[ ]:


X = task_1_data.drop(columns=['mba_p'])
y = task_1_data['mba_p']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1001)


# In[ ]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()


# In[ ]:


reg.fit(X_train, y_train)


# In[ ]:


importance = reg.coef_
for i,v in enumerate(importance):
    print('Feature:',X_train.columns[i],', Score: %.5f' % (v))


# In[ ]:


from sklearn.metrics import r2_score

pred = reg.predict(X_test)
accuracy = r2_score(pred, y_test)

print("Intercept: ",reg.intercept_)
print("Coefficients: ",reg.coef_)
print("R2 Score: ",accuracy)

print("\nRegression Equation: Y = ",reg.coef_[0],"x^2 + ",reg.coef_[1],"x + (",reg.intercept_,")")


# # 2. Regression using Ensemble Models, to predict future salaries

# In[ ]:


#Dropping non-numerical columns and primary keys

task_2_data = placement.drop(columns=['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status','sl_no','etest_p'])

#Dropping null values

task_2_data.dropna(inplace=True)

#View modified data

task_2_data


# In[ ]:


X = task_2_data.drop(columns = ['salary'])
y = task_2_data['salary']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1001)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge

xgb = XGBRegressor(learning_rate=0.01, n_estimators=1000)
clf = RandomForestRegressor(n_estimators=1000)
gb = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
lgbm = LGBMRegressor(boosting_type='gbdt', objective='huber', learning_rate=0.01, n_estimators=1000, metric='rmse')
lassoreg = Lasso(alpha=0.1, normalize=True)
ridgereg = Ridge(alpha=0.1, normalize=True)


# In[ ]:


xgb_scores = cross_val_score(xgb, X_train, y_train, cv=10)
clf_scores = cross_val_score(clf, X_train, y_train, cv=10)
gb_scores = cross_val_score(gb, X_train, y_train, cv=10)
lgbm_scores = cross_val_score(lgbm, X_train, y_train, cv=10)
lasso_scores = cross_val_score(lassoreg, X_train, y_train, cv=10)
ridge_scores = cross_val_score(ridgereg, X_train, y_train, cv=10)


# In[ ]:


print("XGB Regression: ",np.mean(xgb_scores))
print("Random Forest Regression: ",np.mean(clf_scores))
print("Gradient Boosting Regression: ",np.mean(gb_scores))
print("LGBM Regression: ",np.mean(lgbm_scores))
print("Lasso Regression: ",np.mean(lasso_scores))
print("Ridge Regression: ",np.mean(ridge_scores))


# In[ ]:


#LGBM gave the best results of the cross validation.
#We will use it to predict our salaries.

lgbm.fit(X_train, y_train)
pred_1 = lgbm.predict(X_test)
print(lgbm.score(X_test, y_test))

#We also want to get a picture of the decision tree!
#That's why we're fitting the clf model below.

clf.fit(X_train, y_train)
pred_2 = clf.predict(X_test)
print(clf.score(X_test, y_test))


# In[ ]:


#These are the actual, correct values of the salaries.
#Observe the difference between this table and the one below.

actual_values = pd.DataFrame({'actual_salary':y_test})
actual_values.index = X_test.index

actual_values.to_csv("Actual_Salaries.csv", index=False)

actual_values


# In[ ]:


#Clearly, LGBM Regression gives the best results here. Let's see its predictions.

model_preds = pd.DataFrame({'predicted_salary':pred_1})
model_preds.index = X_test.index

model_preds.to_csv("Predicted_Salaries.csv",index=False)

model_preds


# # Now, we're going to get an image of the decision tree.

# In[ ]:


from sklearn.tree import export_graphviz
import pydot


# In[ ]:


feature_list = list(X_test.columns)
feature_list


# In[ ]:


#Pulling out a tree

tree = clf.estimators_[5]


# In[ ]:


export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)


# In[ ]:


(graph, ) = pydot.graph_from_dot_file('tree.dot')


# In[ ]:


#Export image of tree

graph.write_png('tree.png')

