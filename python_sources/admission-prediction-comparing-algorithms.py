#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy
import xgboost
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


df.pop('Serial No.')
df.hist(figsize=(12,8))
plt.show()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


corr=df.corr()
sns.heatmap(corr, vmax=0.9,vmin=0,annot=True,cmap="YlGnBu")
corr


# In[ ]:


sns.pairplot(df.drop(columns='Research'))


# In[ ]:


df.columns.values


# In[ ]:


X=df.drop(['Chance of Admit ','SOP'],axis=1) #SOP dropeed sue to high p-value
print(X.head())
Y=df.iloc[:,-1]
print(Y.head())


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
cols=X.columns
array=np.asarray(X[cols])
rescaledX = scaler.fit_transform(array)
rescaledX


# In[ ]:


X= pd.DataFrame(data=rescaledX,columns=cols)
X.head()


# In[ ]:


import statsmodels.api as sm
mod = sm.OLS(Y,X)
res=mod.fit()
print(res.summary())


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=50)


# **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
predictions=reg.predict(X_test)
from sklearn.metrics import r2_score
R2=r2_score(Y_test,predictions)
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y_test,predictions)
print(f'R-square= {round(R2*100,2)}% \nMean Squared Error= {"%.10f" %MSE}')


# **Support Vector Regressor**

# In[ ]:


from sklearn.svm import SVR
clf = SVR()
clf.fit(X_train, Y_train)
predictions=clf.predict(X_test)
predictions=clf.predict(X_test)
from sklearn.metrics import r2_score
R2=r2_score(Y_test,predictions)
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y_test,predictions)
print(f'R-square= {round(R2*100,2)}% \nMean Squared Error= {"%.10f" %MSE}')


# **XGB Regressor**

# In[ ]:


from xgboost import XGBRegressor
XGBreg=XGBRegressor()
XGBreg.fit(X_train,Y_train)
xgboost.plot_importance(XGBreg)
plt.show()


# In[ ]:


predictions=XGBreg.predict(X_test)
from sklearn.metrics import r2_score
R2=r2_score(Y_test,predictions)
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y_test,predictions)
print(f'R-square= {round(R2*100,2)}% \nMean Squared Error= {"%.10f" %MSE}')


# **Random Forest Regressor**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfr.fit(X_train,Y_train)
predictions = rfr.predict(X_test) 
from sklearn.metrics import r2_score
R2=r2_score(Y_test,predictions)
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(Y_test,predictions)
print(f'R-square= {round(R2*100,2)}% \nMean Squared Error= {"%.10f" %MSE}')


# In[ ]:




