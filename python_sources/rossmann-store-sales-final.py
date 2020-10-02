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

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
test=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')


# In[ ]:


print(train.shape)
print(test.shape)
print(store.shape)


# In[ ]:


train.head()


# In[ ]:


store.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train['StateHoliday'].value_counts()


# In[ ]:


train.describe()[['Sales','Customers']]


# In[ ]:


train.describe()[['Sales','Customers']].loc['mean']


# In[ ]:


train.describe()[['Sales','Customers']].loc['min']


# In[ ]:


train.describe()[['Sales','Customers']].loc['max']


# In[ ]:


train['Store'].value_counts().head(20)


# In[ ]:


train['Store'].value_counts().tail(20)


# In[ ]:


train['DayOfWeek'].value_counts()


# In[ ]:


train['Open'].value_counts()


# In[ ]:


train['Promo'].value_counts()


# In[ ]:


train['Date']=pd.to_datetime(train['Date'],format='%Y-%m-%d')


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


store.isna().sum()


# In[ ]:


store1=train[train['Store']==1]


# In[ ]:


store1.head()


# In[ ]:


store1.shape


# In[ ]:


store1.resample('1d',on='Date')['Sales'].sum().plot.line(figsize=(15,5))


# In[ ]:


store1[store1['Sales']==0]


# In[ ]:


test_store1=test[test['Store']==1]
test_store1['Date']=pd.to_datetime(test_store1['Date'],format='%Y-%m-%d')


# In[ ]:


test_store1['Date'].min(),test_store1['Date'].max()


# In[ ]:


test_store1['Open'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.hist(store1['Sales'])


# In[ ]:


store[store['Store']==1].T


# In[ ]:


store[~store['Promo2SinceYear'].isna()].iloc[0]


# #### MISSING VALUE TREATMENT

# In[ ]:


store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')


# In[ ]:


store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)


# In[ ]:


store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])


# In[ ]:


store['PromoInterval']=store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])


# In[ ]:


store['CompetitionDistance']=store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])
store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])


# In[ ]:


store.isna().sum()


# ### MERGING DATA

# In[ ]:


df=train.merge(store,on='Store',how='left')


# In[ ]:


print(train.shape)
print(df.shape)


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# ### Encoding
# ##### 3 categorical cols, 1 date coln, rest are numerical
# 

# In[ ]:


df['day']=df['Date'].dt.day
df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month


# In[ ]:


#df['Date'].dt.strftime('%a')


# In[ ]:


#Dummies: StateHoliday,StoreType,Assortment,PromoInterval

df['StateHoliday']=df['StateHoliday'].apply(lambda x:'0' if x==0 or x=='0' else x)


# In[ ]:


df['StateHoliday']=df['StateHoliday'].map({'0':0,'a':1,'b':2,'c':3})
df['StateHoliday']=df['StateHoliday'].astype(int)


# In[ ]:


df['StoreType'].value_counts()


# In[ ]:


df['StoreType']=df['StoreType'].map({'a':0,'b':1,'c':2,'d':3})


# In[ ]:


df['Assortment']=df['Assortment'].map({'a':0,'b':1,'c':2})


# In[ ]:


df['Assortment']=df['Assortment'].astype(int)


# In[ ]:


df['PromoInterval'].value_counts()


# In[ ]:


df['PromoInterval']=df['PromoInterval'].map({'Jan,Apr,Jul,Oct':0,'Feb,May,Aug,Nov':1,'Mar,Jun,Sept,Dec':2})


# In[ ]:


df['PromoInterval']=df['PromoInterval'].astype(int)


# In[ ]:


df=df.drop('Date',1)


# In[ ]:


df.dtypes


# In[ ]:


y=np.log1p(df['Sales'])


# In[ ]:


X=df.drop(['Sales','Customers'],1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=1,test_size=0.2)


# In[ ]:


y.plot.hist()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor(max_depth=11,random_state=1).fit(X_train,y_train)


# In[ ]:


y_pred_val=dt.predict(X_val)


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error

print(r2_score(y_val,y_pred_val))
print(np.sqrt(mean_squared_error(y_val,y_pred_val)))


# In[ ]:


y_val_exp=np.exp(y_val)-1
y_pred_val_exp=np.exp(y_pred_val)-1
np.sqrt(mean_squared_error(y_val_exp,y_pred_val_exp))


# In[ ]:


r2_score(y_val_exp,y_pred_val_exp)


# In[ ]:


get_ipython().system('pip install pydotplus')


# In[ ]:


def draw_tree(model, columns):
    import pydotplus
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    import os
    from sklearn import tree
    
    graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + graphviz_path

    dot_data = StringIO()
    tree.export_graphviz(model,
                         out_file=dot_data,
                         feature_names=columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())


# In[ ]:


draw_tree(dt,X_train.columns)


# ### Preprocessing
# - Missing value treatment
# - Merging train with store file
# - Log transformation of target feature
# - Exponential transformation after prediction
# - Make sure columns are in both train and test

# In[ ]:


print(dt.feature_importances_)
plt.barh(y=X_train.columns,width=dt.feature_importances_)
plt.show()


# In[ ]:


test.head()


# In[ ]:


avg_cust=df.groupby(['Store'])[['Customers']].mean().astype(int)
test1=test.merge(avg_cust,on='Store',how='left')


# In[ ]:


test.shape,test1.shape


# In[ ]:


test_merged=test1.merge(store,on='Store',how='left')


# In[ ]:


test1.shape,test_merged.shape


# In[ ]:


test_merged.head()


# In[ ]:


test_merged['Open'].fillna(1,inplace=True)


# In[ ]:


test_merged.isna().sum()


# In[ ]:


test_merged['Date']=pd.to_datetime(test_merged['Date'],format='%Y-%m-%d')
test_merged['day']=test_merged['Date'].dt.day
test_merged['month']=test_merged['Date'].dt.month
test_merged['year']=test_merged['Date'].dt.year


# In[ ]:


test_merged=test_merged.drop('Date',1)


# In[ ]:


test_merged['StateHoliday']=test_merged['StateHoliday'].apply(lambda x:'0' if x==0 or x=='0' else x)


# In[ ]:


test_merged['StateHoliday'].value_counts()


# In[ ]:


test_merged['StoreType'].value_counts()


# In[ ]:


test_merged['Assortment'].value_counts()


# In[ ]:


test_merged['PromoInterval'].value_counts()


# In[ ]:


test_merged['StateHoliday']=test_merged['StateHoliday'].map({'0':0,'a':1})
test_merged['StateHoliday']=test_merged['StateHoliday'].astype(int)

test_merged['StoreType']=test_merged['StoreType'].map({'a':0,'b':1,'c':2,'d':3})
test_merged['StoreType']=test_merged['StoreType'].astype(int)

test_merged['Assortment']=test_merged['Assortment'].map({'a':0,'b':1,'c':2})
test_merged['Assortment']=test_merged['Assortment'].astype(int)

test_merged['PromoInterval']=test_merged['PromoInterval'].map({'Jan,Apr,Jul,Oct':0,'Feb,May,Aug,Nov':1,'Mar,Jun,Sept,Dec':2})
test_merged['PromoInterval']=test_merged['PromoInterval'].astype(int)


# In[ ]:


test_merged1=test_merged.drop('Id',1)


# In[ ]:


test_merged1.head()


# In[ ]:


X_train.head()


# In[ ]:


test_merged1.shape


# In[ ]:


y_pred=dt.predict(test_merged1[X_train.columns])


# In[ ]:


y_pred


# In[ ]:


y_pred_exp=np.exp(y_pred)-1


# In[ ]:


submission_pred=pd.DataFrame(test_merged['Id'],columns=['Id'])


# In[ ]:


submission_pred['Sales']=y_pred_exp


# In[ ]:


submission_pred['Id']=np.arange(1,len(submission_pred)+1)


# In[ ]:


submission_pred


# In[ ]:





# In[ ]:


submission_pred.to_csv('Submission.csv',index=False)


# In[ ]:


# Credit: kaggle.com
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(y, yhat):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

rmse_val=np.sqrt(mean_squared_error(y_val_exp,y_pred_val_exp))
rmspe_val=rmspe(y_val_exp,y_pred_val_exp)
print(rmse_val,rmspe_val)


# In[ ]:


from sklearn.model_selection import GridSearchCV
params={'max_depth':list(range(5,20))}
base_model=DecisionTreeRegressor()
cv_model=GridSearchCV(base_model,param_grid=params,return_train_score=True).fit(X_train,y_train)


# In[ ]:


df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)


# In[ ]:


df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()
df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()
plt.show()


# In[ ]:


df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)[['param_max_depth','mean_test_score','mean_train_score']]
df_cv_results


# In[ ]:


def get_rmspe_score(model,input_values,y_actual):
    y_predicted=model.predict(input_values)
    y_actual=np.exp(y_actual)-1
    y_predicted=np.exp(y_predicted)-1
    score=rmspe(y_actual,y_predicted)
    return score

params={'max_depth':list(range(5,8))}
base_model=DecisionTreeRegressor()
cv_model=GridSearchCV(base_model,param_grid=params,return_train_score=True,scoring=get_rmspe_score).fit(X_train,y_train)
pd.DataFrame(cv_model.cv_results_)[['params','mean_test_score','mean_train_score']]


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

model_ada=AdaBoostRegressor(n_estimators=5).fit(X_train,y_train)


# In[ ]:


model_ada.estimators_[0]


# In[ ]:


def draw_tree(model, columns):
    import pydotplus
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    import os
    from sklearn import tree
    
    graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'
    os.environ["PATH"] += os.pathsep + graphviz_path

    dot_data = StringIO()
    tree.export_graphviz(model,
                         out_file=dot_data,
                         feature_names=columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())


# In[ ]:


features=X_train.columns
draw_tree(model_ada.estimators_[0],features)


# In[ ]:


import xgboost as xgb


# In[ ]:


dtrain=xgb.DMatrix(X_train,y_train)
dvalidate=xgb.DMatrix(X_val,y_val)

param={'max_depth':5,'eta':1,'ojective':'reg:linear'}
model_xg=xgb.train(param,dtrain,200)
pred_y=model_xg.predict(dvalidate)


val_y_inv=np.exp(y_val)-1
pred_y_inv=np.exp(pred_y)-1
rmspe_val=rmspe(val_y_inv,pred_y_inv)
print(rmspe_val)


# In[ ]:


test_merged=test_merged.drop(['Id','Customers'],1)


# In[ ]:


y_pred_xg=model_xg.predict(xgb.DMatrix(test_merged[X_train.columns]))


# In[ ]:


y_pred_xg_exp=np.exp(y_pred_xg)-1


# In[ ]:


y_pred_xg_exp


# In[ ]:


submission_predicted1 = pd.DataFrame({'Id': test['Id'], 'Sales': y_pred_xg_exp})
testop0=(test[test['Open']==0]['Open']).index
Sales1=[]
for i in range(41088):
    if i in testop0:
        Sales1.append(0)
    else:
        Sales1.append(submission_predicted1['Sales'][i])
submission_predicted1['Sales']=Sales1
print(submission_predicted1.head())
submission_predicted1.to_csv('submission.csv', index=False)


# In[ ]:




