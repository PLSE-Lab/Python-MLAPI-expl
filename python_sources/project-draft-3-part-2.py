#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


epl = pd.read_csv('/kaggle/input/newepl/newEPl (1).csv')
epl


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='HomeTeam')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='AwayTeam')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PFTHG')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PFTAG')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PFTG')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PFTR')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PHTHG')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PHTAG')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PHTG')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='PHTR')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='HTHG')


# In[ ]:


epl.plot(kind='scatter',x='Total_goals',y='HTAG')


# In[ ]:


X_data = epl[['HomeTeam','AwayTeam','HTHG','HTAG']]
Y_data = epl['Total_goals']


# In[ ]:


reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.001)
reg.fit(X_train,y_train)
print("Regression Coefficients")
pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])


# In[ ]:


predicted = reg.predict(X_test)
predicted = predicted.astype(int)
da = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
da


# In[ ]:


regr = RandomForestRegressor(max_depth=20, random_state=0,n_estimators=100)
regr.fit(X_train,y_train)
regr.score(X_test,y_test)
predicted2 = regr.predict(X_test)
predicted2 = predicted2.astype(int)
d = pd.DataFrame({'Actual': y_test, 'Predicted': predicted2})
d


# In[ ]:


regn = MLPRegressor(hidden_layer_sizes=(100,100,100,100))
regn.fit(X_train,y_train)
test_predicted = reg.predict(X_test)
test_predicted = test_predicted.astype(int)
n = pd.DataFrame({'Actual': y_test, 'Predicted': test_predicted})
n


# In[ ]:


n.plot(kind='bar')


# https://drive.google.com/drive/folders/1-YUqkxSeH8ozhxoQswB7hiJs2WbRVeTF

# In[ ]:


a=32 #home team
b=11 #Away team
c=3 #HTHG
d=2 #HTAG
q = {'home team':[a],'away team':[b],'HTHG':[c],'HTAG':[d]}
new=pd.DataFrame(q)
reg.predict(new)


# In[ ]:


a=32 #home team
b=11 #Away team
c=3 #HTHG
d=2 #HTAG
q = {'home team':[a],'away team':[b],'HTHG':[c],'HTAG':[d]}
new=pd.DataFrame(q)
regn.predict(new)

