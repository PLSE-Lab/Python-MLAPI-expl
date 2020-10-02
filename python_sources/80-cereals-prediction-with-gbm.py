#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv("../input/80-cereals/cereal.csv").copy()


# In[ ]:


df.head()


# In[ ]:


df.name.unique()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe().T


# In[ ]:


dummies = df['name'].str.get_dummies()


# In[ ]:


dummies


# In[ ]:


df = pd.concat([df,dummies],axis=1)


# In[ ]:


df.head()


# In[ ]:


df=df.drop(['name'],axis=1)


# In[ ]:


dummies_mfr = df['mfr'].str.get_dummies()


# In[ ]:


df=pd.concat([df,dummies_mfr],axis=1)


# In[ ]:


df = df.drop(['mfr'],axis=1)


# In[ ]:


df.head()


# In[ ]:


dummies_c = df['type'].str.get_dummies()


# In[ ]:


df = pd.concat([df,dummies_c],axis=1)


# In[ ]:


df = df.drop(['type'],axis=1)


# In[ ]:


df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


y = df.rating
x = df.drop('rating',axis=1)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# In[ ]:


gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train, y_train)


# PREDICTION

# In[ ]:


y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


gbm_model.score(X_test,y_test)


# MODEL TUNING

# In[ ]:


gbm_params = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [3, 5, 8,50,100],
    'n_estimators': [200, 500, 1000, 2000],
    'subsample': [1,0.5,0.75],
}


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score


# In[ ]:


gbm = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv_model.fit(X_train, y_train)


# In[ ]:


gbm_cv_model.best_params_


# In[ ]:


gbm_tuned = GradientBoostingRegressor(learning_rate = 0.1,  
                                      max_depth = 50, 
                                      n_estimators = 1000, 
                                      subsample = 0.5)

gbm_tuned = gbm_tuned.fit(X_train,y_train)


# In[ ]:


y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


gbm_tuned.score(X_test, y_test)#test accuracy score

