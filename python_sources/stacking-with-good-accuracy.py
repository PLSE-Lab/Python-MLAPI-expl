#!/usr/bin/env python
# coding: utf-8

# # For best reults i used Stacking regression of XGBOOST and Randomforest which is demonstrated below...

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


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
submit = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


print("length of train", len(train))
print("length of test", len(test))
print("length of submit", len(submit))


# In[ ]:


train.describe()


# In[ ]:


train = train.drop(['County','Province_State','Country_Region','Target'],axis=1) 
test = test.drop(['County','Province_State','Country_Region','Target'],axis=1)
train.head()


# In[ ]:


test_date_min = test['Date'].min()
test_date_max = test['Date'].max()
train['Date']=pd.to_datetime(train['Date'])
test['Date']=pd.to_datetime(test['Date'])
test['Date']=test['Date'].dt.strftime("%Y%m%d")
train['Date']=train['Date'].dt.strftime("%Y%m%d").astype(int)


# In[ ]:


test.drop(['ForecastId'],axis=1,inplace=True)
test.index.name = 'Id'
test.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold


# In[ ]:


X = train.drop(['TargetValue', 'Id'], axis=1)
y = train["TargetValue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)


# ## Random forest

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# params1={
#  "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#  "n_estimators"     : [100,300,500,700,800,1000],
# }
model = RandomForestRegressor(n_estimators=800,n_jobs=-1,max_depth=10)
# random_search0=RandomizedSearchCV(model,param_distributions=params1,n_iter=5,n_jobs=-1,cv=5,verbose=3)
model.fit(X_train, y_train)


from sklearn.metrics import r2_score
# y_pred2 = model.predict(X_test)
print(r2_score(y_test,model.predict(X_test)))


# ## XGboost

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
reg =xgboost.XGBRegressor(n_estimators=800,n_jobs=-1)

# params={
#  "learning_rate"    : [0.10, 0.25,0.50] ,
#  "max_depth"        : [ 5, 8, 10],
#  "gamma"            : [ 0.0, 0.1, 0.3, 0.4 ],
#  "n_estimators"     : [400,700,1000]
# }

# random_search=RandomizedSearchCV(reg,param_distributions=params,n_iter=5,n_jobs=-1,cv=5,verbose=3)
reg.fit(X_train, y_train)

print(r2_score(y_test,reg.predict(X_test)))


# ## Stacking Xgboost and random forest

# In[ ]:


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LogisticRegression

estimators = [('c1',model),('c2',reg)]

reg_stack = StackingRegressor(
     estimators=estimators,n_jobs=-1
 )

reg_stack.fit(X_train, y_train)

print(r2_score(y_test,reg_stack.predict(X_test)))


# In[ ]:


predictions = reg_stack.predict(test)

pred_list = [int(x) for x in predictions]

output = pd.DataFrame({'Id': test.index, 'TargetValue': pred_list})
print(output)


# In[ ]:


a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()


# In[ ]:


a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05'].clip(0,10000)
a['q0.5']=a['q0.5'].clip(0,10000)
a['q0.95']=a['q0.95'].clip(0,10000)
a


# In[ ]:


a['Id'] =a['Id']+ 1
a


# In[ ]:


sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()

