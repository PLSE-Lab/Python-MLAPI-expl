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


import pandas as pd


# In[ ]:


covid_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
covid_train.head()


# In[ ]:


covid_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
covid_test.head()


# Dataframe Details

# In[ ]:


covid_train.shape


# In[ ]:


covid_test.shape


# In[ ]:


covid_train.columns


# In[ ]:


covid_test.columns


# In[ ]:


covid_train.count()


# In[ ]:


covid_test.count()


# In[ ]:


covid_train.describe()


# In[ ]:


covid_test.describe()


# In[ ]:


covid_train.isna().sum()


# In[ ]:


covid_test.isna().sum()


# In[ ]:


covid_train.isna().any()


# In[ ]:


print(covid_train.isnull().sum())


# In[ ]:


# Rename the Columns of Train and Test Datasets
covid_train.rename(columns={'Country_Region':'Country'}, inplace=True)
covid_test.rename(columns={'Country_Region':'Country'}, inplace=True)

covid_train.rename(columns={'Province_State':'State'}, inplace=True)
covid_test.rename(columns={'Province_State':'State'}, inplace=True)


# In[ ]:


covid_train.info()


# In[ ]:


covid_test.info()


# In[ ]:


covid_train[covid_train['Country']=='China'].groupby(by='Country').sum()


# In[ ]:


covid_train.loc[covid_train.Country == 'China', :].head(10)


# In[ ]:


covid_train[~covid_train['State'].isnull()]['Country'].value_counts()


# In[ ]:


covid_train.corr()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.heatmap(covid_train.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=2)


# In[ ]:


covid_train.boxplot(column=['ConfirmedCases', 'Fatalities','Id'])


# In[ ]:


covid_train.hist()


# In[ ]:


# Converting Date Time object
import time
from datetime import datetime


# In[ ]:


covid_train['Date']=pd.to_datetime(covid_train['Date'],infer_datetime_format=True)
covid_test['Date']=pd.to_datetime(covid_test['Date'],infer_datetime_format=True)


# In[ ]:


covid_train.info()


# In[ ]:


covid_test.info()


# In[ ]:


Groupby = covid_train.groupby(by='Country')[['ConfirmedCases','Fatalities']].sum().reset_index()
plt.figure(figsize=(20,10))

sns.barplot(x='ConfirmedCases',y='Country',data=Groupby[Groupby['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(20))


# In[ ]:


Groupby[Groupby['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(10)


# ### Linear Regression

# In[ ]:


covid_x=covid_train.drop(['ConfirmedCases','Country','State','Date'],axis='columns')
covid_x.head()


# In[ ]:


covid_y=pd.DataFrame(covid_train.iloc[:,-2])
covid_y.head()


# In[ ]:


# Splitting X and y into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(covid_x,covid_y)


# In[ ]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,Y_train)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


tree_regressor=DecisionTreeRegressor()
tree_regressor.fit(X_train,Y_train)


# In[ ]:


y_pred_tree=tree_regressor.predict(X_test)
y_tree_pred_df=pd.DataFrame(y_pred_tree,columns=['Predict_tree'])
y_tree_pred_df.head()


# In[ ]:


DTCscore = tree_regressor.score(X_train,Y_train)
print("Decision Tree Score: ",DTCscore)


# In[ ]:


sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
sub.to_csv('submission.csv',index=False)


# In[ ]:


sns.pairplot(covid_train)


# In[ ]:


sns.distplot(covid_train['ConfirmedCases'], kde=True, rug=True)

