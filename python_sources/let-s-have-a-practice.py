#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats


# In[28]:


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[29]:


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')


# In[30]:


train['price'] = np.log1p(train['price'])


# In[31]:


columns = ['sqft_living','sqft_above','sqft_basement','sqft_lot']

for c in columns:
    train[c] = np.log1p(train[c])
    test[c] = np.log1p(test[c])


# In[32]:


for df in [train,test]:
    df['date'] = df['date'].apply(lambda x: x[0:6]).astype(int)


# In[33]:


train_id = train['id']
train_price = train['price']

del train['id']
del train['price']


# In[34]:


test_id = test['id']
test = test.drop(columns='id')


# In[35]:


print(train.dtypes)


# In[20]:


lgbm = LGBMRegressor(n_estimators=300,learning_rate=0.1)
lgbm.fit(train,train_price)
xgb = XGBRegressor(n_estimators=400,learning_rate=0.3, max_depth=3)
xgb.fit(train,train_price)
gbm = GradientBoostingRegressor(random_state=0)
gbm.fit(train,train_price)


# In[11]:


from sklearn.model_selection import KFold


# In[ ]:


# def get_stacking_base_datasets(model, X_train_n,y_train_n,X_test_n,n_folds):
#     kf = KFold(n_splits=n_folds, shuffle = False, random_state=0)
#     train_fold_pred = np.zeros((X_train_n.shape[0],1))
#     test_pred = np.zeros((X_test_n.shape[0], n_folds))
    
#     for folder_counter, (train_index, valid_index) in enumerate(kf.split(X_train_n)):
#         X_tr = X_train_n[train_index]
#         y_tr = y_train_n[train_index]
#         X_te = X_train_n[valid_index]
#         model.fit(X_tr, y_tr)
#         train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1)
#         test_pred[:, folder_counter] = model.predict(X_test_n)
#     test_pred_mean = np.mean(test_pred, axis=1).reshape(-1,1)
#     return train_fold_pred, test_pred_mean


# In[ ]:


# X_train_n = train.values
# X_test_n = test.values
# y_train_n = train_price.values


# In[ ]:


# lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm, X_train_n, y_train_n, X_test_n, 5)
# gbm_train, gbm_test = get_stacking_base_datasets(gbm, X_train_n, y_train_n, X_test_n, 5)
# xgb_train, xgb_test = get_stacking_base_datasets(xgb, X_train_n, y_train_n, X_test_n, 5)


# In[ ]:


# Stack_final_X_train = np.concatenate((lgbm_train, gbm_train, xgb_train),axis=1)
# Stack_final_X_test = np.concatenate((lgbm_test,gbm_test,xgb_test),axis=1)


# In[36]:


lgbm_pred = (lgbm.predict(test))
xgb_pred = (xgb.predict(test))



# In[37]:


pred = np.expm1(0.5*lgbm_pred + 0.5*xgb_pred)


# In[ ]:


# pred = np.array([lgbm_pred,xgb_pred,gbm_pred])
# print(pred.shape)
# pred = np.transpose(pred)
# print(pred.shape)


# In[ ]:


# xgb_final = XGBRegressor(n_estimators=400,learning_rate=0.1, max_depth=3)
# lgbm_final=LGBMRegressor(n_estimators=400)


# In[ ]:


# xgb_final.fit(Stack_final_X_train,train_price)
# lgbm_final.fit(Stack_final_X_train,train_price)


# In[ ]:


# final_xgb = np.expm1(xgb_final.predict(Stack_final_X_test))
# final_lgbm = np.expm1(lgbm_final.predict(Stack_final_X_test))


# In[ ]:


# (final_xgb)


# In[38]:


pred


# In[39]:


# submission_lgbm = pd.DataFrame({'id': test_id, 'price': lgbm_pred})
# submission_xgb = pd.DataFrame({'id':test_id, 'price': xgb_pred})
# submission_gbm = pd.DataFrame({'id':test_id, 'price':gbm_pred})
# submission_stacking_xgb = pd.DataFrame({'id':test_id, 'price':final_xgb})
# submission_stacking_lgbm = pd.DataFrame({'id':test_id, 'price':final_lgbm})
submission_new = pd.DataFrame({'id':test_id, 'price':pred})


# In[40]:


# submission_lgbm.to_csv('lgbm_step.csv', index=False)
# submission_xgb.to_csv('xgb_step.csv',index=False)
# submission_gbm.to_csv('gbm_step.csv', index=False)
# submission_stacking_xgb.to_csv('stacking_xgb.csv',index=False)
# submission_stacking_lgbm.to_csv('stacking_lgbm.csv',index=False)
submission_new.to_csv('new.csv',index=False)

