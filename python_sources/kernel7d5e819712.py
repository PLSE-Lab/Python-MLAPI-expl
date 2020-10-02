#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from pandas import Series, DataFrame
data=pd.read_csv("/kaggle/input/OnlineNewsPopularity.csv")
df=pd.DataFrame(data)
df.head()


# **DATA EXPLORE**

# In[ ]:


df.shape


# In[ ]:


all_columns=list(df.columns)
categorical_cols = all_columns[13:19] + all_columns[31:39]
categorical_cols


# In[ ]:


df.describe()


# In[ ]:


df[' shares'].max()


# In[ ]:


df.hist(figsize=(20,20))
plt.show()


# **outlier analysis**

# In[ ]:


figsize=plt.rcParams["figure.figsize"]
figsize[0]=10
figsize[1]=10
plt.hist(x=df[' shares'])


# outliers present evident from above graph
# Removing outliers

# In[ ]:


u=df[' shares'].median()
s=df[' shares'].std()


# removing tuples having value higer the median + 2* std and less tha median-2*s

# In[ ]:


df=df[df[' shares']<(u+2*s) ]
df.shape


# **missing values detect**

# In[ ]:


df.isna().sum()


# Here the missing values are replaced by 0. n_token_content is 0 in various tuples which is not possible. So,removing those tuples

# finding no of 0 in each column

# In[ ]:


for i in df.columns:
    c=0
    for j in df.index:
        if(df[i][j]==0):
            c=c+1
    if(c!=0):
        print(i,c)


# In[ ]:


for i in df.index:
    if(df[' n_tokens_content'][i]==0):
        df=df.drop(i,axis=0)   


# In[ ]:


df.shape


# In[ ]:





# **dimension reduction**

# method 1 (low variance filter)

# In[ ]:


df.var().sort_values()


# method 2 (high correlation filter)
# using this method  here to reduce dimension

# In[ ]:


corrmat=df.corr()
df.corr()[' shares'].sort_values(ascending=False)


# In[ ]:


reduced_column= df.corr()[' shares'].nsmallest(35)
reduced_column


# In[ ]:


df.shape


# method 3(using random forest feature importance)

# In[ ]:


model=RandomForestRegressor(random_state=42)
dx=df.drop(['url',' timedelta',' shares'],axis=1);
dy=df[' shares']
model.fit(dx,dy)


# In[ ]:


figsize=plt.rcParams['figure.figsize']
figsize[0]=10
figsize[1]=10


# In[ ]:


feature=pd.Series(model.feature_importances_,index=dx.columns)
columns=df.columns
feature.nlargest(30).plot(kind='barh')


# In[ ]:


plt.barh(range(len(feature)),feature)
plt.yticks(range(len(feature)),columns)


# pre processed data

# In[ ]:


X=df.drop(['url',' timedelta',' shares'],axis=1)
for i in X.columns:
    if( corrmat[' shares'][i]<= 0.00689):
        X=X.drop(i,axis=1)
print(X.shape)
Y=df[' shares']
x_train,x_cv,y_train,y_cv= train_test_split(X,Y,test_size=0.2,random_state=42)
some_x_cv=x_cv.iloc[:500]
some_y_cv=y_cv.iloc[:500]
some_x_data=x_train.iloc[:500]
some_y_data=y_train.iloc[:500]


# **LINEAR REGRESSION**

# In[ ]:


lreg=LinearRegression()
lreg.fit(x_train,y_train)
lreg.coef_


# In[ ]:


lreg.intercept_


# for train data

# In[ ]:


pred=lreg.predict(some_x_data)
model1_test= pd.DataFrame({'Actual': some_y_data.values.flatten(),'Predicted': pred.flatten()})
model1_test


# In[ ]:


lin_mse = mean_squared_error(some_y_data,pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


lin_mae=mean_absolute_error(some_y_data,pred)
lin_mae


# In[ ]:


a,b=plt.subplots()
sns.regplot(x=some_y_data,y=pred)


# for test data

# In[ ]:


pred=lreg.predict(some_x_cv)
model1_test= pd.DataFrame({'Actual': some_y_cv.values.flatten(),'Predicted': pred.flatten()})
model1_test


# In[ ]:


lin_mse = mean_squared_error(some_y_cv,pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


lin_mae=mean_absolute_error(some_y_cv,pred)
lin_mae


# In[ ]:


a,b=plt.subplots()
sns.regplot(x=some_y_cv,y=pred)


# **DECISION TREE**

# In[ ]:


model2= DecisionTreeRegressor(random_state=42)
model2.fit(x_train,y_train)


# for train data

# In[ ]:


model2_pred=model2.predict(some_x_data)


# In[ ]:


model2_check= pd.DataFrame({'Actual': some_y_data.values.flatten(),'Predicted': model2_pred.flatten()})
model2_check


# In[ ]:


model2_mse = mean_squared_error(some_y_data,model2_pred)
model2_rmse = np.sqrt(model2_mse)
model2_rmse


# In[ ]:


a,b=plt.subplots(figsize=(17,4))
sns.regplot(x=some_y_data,y=model2_pred)


# for test data

# In[ ]:


model2_test_pred=model2.predict(some_x_cv)


# In[ ]:


model2_test_check= pd.DataFrame({'Actual': some_y_cv.values.flatten(),'Predicted': model2_test_pred.flatten()})
model2_test_check


# In[ ]:


model2_test_mse = mean_squared_error(some_y_cv,model2_test_pred)
model2_test_rmse = np.sqrt(model2_test_mse)
model2_test_rmse


# In[ ]:


a,b=plt.subplots(figsize=(17,4))
sns.regplot(x=some_y_cv,y=model2_pred)


# **RANDOM FOREST**

# In[ ]:


model3= RandomForestRegressor(random_state=42)
model3.fit(x_train,y_train)


# for train data

# In[ ]:


model3_pred=model3.predict(some_x_data)
model3_check= pd.DataFrame({'Actual': some_y_data.values.flatten(),'Predicted': model3_pred.flatten()})
model3_check


# In[ ]:


rf_mse = mean_squared_error(some_y_data,model3_pred)
rf_rmse = np.sqrt(rf_mse)
rf_rmse


# In[ ]:


a,b=plt.subplots(figsize=(17,4))
sns.regplot(x=some_y_data,y=model3_pred)


# for test data

# In[ ]:


model3_pred=model3.predict(some_x_cv)
model3_check= pd.DataFrame({'Actual': some_y_cv.values.flatten(),'Predicted': model3_pred.flatten()})
model3_check


# In[ ]:


rf_mse = mean_squared_error(some_y_cv,model3_pred)
rf_rmse = np.sqrt(rf_mse)
rf_rmse


# In[ ]:


a,b=plt.subplots(figsize=(17,4))
sns.regplot(x=some_y_cv,y=model3_pred)


# In[ ]:


from sklearn.metrics import r2_score

r2_score( some_y_cv, model3_pred)


# using k fold cross validation decision tree

# In[ ]:


score_tree=cross_val_score(model2,x_train,y_train,scoring="neg_mean_squared_error",cv=10)
tree_rmse=np.sqrt(-score_tree)
tree_rmse


# In[ ]:


tree_rmse.mean()


# using k fold cross validation for linear regression 

# In[ ]:


score_linear=cross_val_score(lreg,x_train,y_train,scoring="neg_mean_squared_error",cv=20)
linear_rmse=np.sqrt(-score_tree)
linear_rmse


# In[ ]:


linear_rmse.mean()


# using k cross validation random forest

# In[ ]:


score_linear=cross_val_score(model3,x_train,y_train,scoring="neg_mean_squared_error",cv=20)
rfd_rmse=np.sqrt(-score_tree)
rfd_rmse


# In[ ]:


rfd_rmse.mean()

