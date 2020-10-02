#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.nunique()


# In[ ]:


df['enginetype'].unique()


# In[ ]:


df['cylindernumber'].unique()


# In[ ]:


df['enginetype'].value_counts()


# In[ ]:


df['cylindernumber']=df['cylindernumber'].map({'four':4,'six':6,'five':5,'eight':8,'two':2,'three':3,'twelve':12})


# In[ ]:


df['cylindernumber'].dtype


# In[ ]:


df['enginetype']=df['enginetype'].map({'ohc':1,'ohcf':2,'ohcv':3,'dohc':4,'l':5,'rotar':6,'dohcv':7})


# In[ ]:


sns.boxplot(data=df,x='cylindernumber',y='price')


# In[ ]:


sns.boxplot(data=df,x='enginelocation',y='price')


# In[ ]:


sns.boxplot(data=df,x='carbody',y='price')


# In[ ]:


plt.scatter(x='compressionratio',y='price',data=df)


# In[ ]:


df.head(1)


# In[ ]:


sns.boxplot(data=df,x='symboling',y='price')


# In[ ]:


sns.scatterplot(x='price',y='curbweight',hue='doornumber',data=df)


# In[ ]:


plt.rcParams['figure.figsize']=(12,12)
corr=df.corr()
sns.heatmap(corr,fmt='.2f',annot=True,cmap=plt.cm.Blues)


# In[ ]:


df_corr=df.corr().abs()
df_corr


# In[ ]:


upper=df_corr.where(np.triu(np.ones(df_corr.shape),k=1).astype(np.bool))
upper


# In[ ]:


to_drop=[column for column in upper.columns if any (upper[column]>0.95)]
print('----------------------------')
print(to_drop)


# In[ ]:


df1=df.drop(to_drop,axis=1)
df1.columns


# In[ ]:


df1.drop('car_ID',axis=1,inplace=True)


# In[ ]:


df1.info()


# In[ ]:


df1.nunique()


# In[ ]:


dummies=pd.get_dummies(df1[['fueltype','aspiration','doornumber','carbody','drivewheel',
                            'enginelocation','fuelsystem']])


# In[ ]:


dummies.head(2)


# In[ ]:


df1=pd.concat([df1,dummies],axis=1)


# In[ ]:


df1.drop(['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','fuelsystem'],
         axis=1,inplace=True)


# In[ ]:


#checking for null values in enginetype feature
df1[df1['enginetype'].isnull()]


# In[ ]:


#filling NaN values with most common enginetype which is 1
df1['enginetype']=df1['enginetype'].fillna('1')


# In[ ]:


df1[df1['enginetype'].isnull()]


# In[ ]:


df1['enginetype']=df1['enginetype'].astype('int')


# In[ ]:


df1.info()


# In[ ]:


X=df1.loc[:,df1.columns!='price']
y=df1.loc[:,'price']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)


# In[ ]:


X_train.drop('CarName',axis=1,inplace=True)
X_test.drop('CarName',axis=1,inplace=True)


# In[ ]:





# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


lr.score(X_train,y_train)


# In[ ]:


lr_pred=lr.predict(X_test)
print('MSE:',mean_squared_error(lr_pred,y_test))
print('MAE:',mean_absolute_error(lr_pred,y_test))
print('r2_score:',r2_score(lr_pred,y_test))


# In[ ]:


prediction=pd.DataFrame({'Actual':y_test,'Predicted':lr_pred})


# In[ ]:


prediction.head(10)


# In[ ]:


lr.coef_


# # SVM

# In[ ]:


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# In[ ]:


svc=SVR()
svc.fit(X_train,y_train)


# In[ ]:


svc.score(X_train,y_train)


# In[ ]:


svc_pred=svc.predict(X_test)
print('MSE:',mean_squared_error(svc_pred,y_test))
print('MAE:',mean_absolute_error(svc_pred,y_test))
print('r2_score:',r2_score(svc_pred,y_test))


# ### - Scaling the data before training by SVR

# In[ ]:


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[ ]:


svr=SVR()
svr.fit(X_train_scaled,y_train)


# In[ ]:


svr.score(X_train_scaled,y_train)


# In[ ]:


svr_pred=svr.predict(X_test_scaled)


# In[ ]:


print('MSE:',mean_squared_error(svr_pred,y_test))
print('MAE:',mean_absolute_error(svr_pred,y_test))
print('r2_score:',r2_score(svr_pred,y_test))


# # Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf=RandomForestRegressor()
rf.fit(X_train,y_train)


# In[ ]:


rf.score(X_train,y_train)


# In[ ]:


rf_pred=rf.predict(X_test)
print('MSE:',mean_squared_error(rf_pred,y_test))
print('MAE:',mean_absolute_error(rf_pred,y_test))
print('r2_score:',r2_score(rf_pred,y_test))


# In[ ]:


prediction_rf=pd.DataFrame({'Actual':y_test,'Predicted':rf_pred})
prediction_rf.head(10)


# # Bagging Regressor

# In[ ]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


bag=BaggingRegressor()
bag.fit(X_train,y_train)
bag.score(X_train,y_train)


# In[ ]:


bag_pred=bag.predict(X_test)
print('MSE:',mean_squared_error(bag_pred,y_test))
print('MAE:',mean_absolute_error(bag_pred,y_test))
print('r2_score:',r2_score(bag_pred,y_test))


# In[ ]:


#Support vector Regressor performs bad and it is less generally used in regression problems
#Linear regression and random forest gives good prediction accuracy on the data


# In[ ]:




