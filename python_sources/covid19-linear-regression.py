#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sb
import matplotlib.pyplot as plt
import time
from datetime import datetime
# Any results you write to the current directory are saved as output.


# In[ ]:


covid_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
covid_train.head()


# In[ ]:


covid_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
covid_test.head()


# In[ ]:


covid_train.shape


# In[ ]:


covid_test.shape


# In[ ]:


covid_train.describe()


# In[ ]:


covid_train.info()


# In[ ]:


covid_train['Date']=pd.to_datetime(covid_train['Date'],infer_datetime_format=True)
covid_test['Date']=pd.to_datetime(covid_test['Date'],infer_datetime_format=True)


# In[ ]:


covid_train.info()


# In[ ]:


covid_test.info()


# In[ ]:


covid_train.hist()


# In[ ]:


covid_test.hist()


# In[ ]:



covid_train.shape


# In[ ]:


corr=covid_train.corr()
sb.heatmap(corr,vmax=1.,square=True)


# In[ ]:


g=sb.heatmap(covid_train[["Id","ConfirmedCases","Fatalities"]].corr(),annot=True,fmt=".2f",cmap="coolwarm")


# In[ ]:


covid_x=pd.DataFrame(covid_train.iloc[:,-1])
covid_x.head()


# In[ ]:


covid_y=pd.DataFrame(covid_train.iloc[:,-2])
covid_y.head()


# Linear Regression and Decision Tree Regressor

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(covid_x,covid_y,test_size=0.3)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


regression=LinearRegression()
regression.fit(X_train,Y_train)
#regression['Date']=regression['Date'].astype(int)
tree_regressor=DecisionTreeRegressor()
tree_regressor.fit(X_train,Y_train)

y_pred_lin=regression.predict(X_test)
y_pred_df=pd.DataFrame(y_pred_lin,columns=['Predict'])
Y_test.head()


# In[ ]:


y_pred_df.head()


# In[ ]:


y_pred_tree=tree_regressor.predict(X_test)
y_tree_pred_df=pd.DataFrame(y_pred_tree,columns=['Predict_tree'])
y_tree_pred_df.head()


# In[ ]:


plt.figure(figsize=(5,5))
plt.title('Actual vs Prediction')
plt.xlabel('Fatalities')
plt.ylabel('Predicted')
plt.legend()
plt.scatter((X_test['Fatalities']),(Y_test['ConfirmedCases']),c='red')
plt.scatter((X_test['Fatalities']),(y_pred_df['Predict']),c='cyan')
plt.show()
            


# In[ ]:


sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
sub.to_csv('submission_csv',index=False)


# In[ ]:




