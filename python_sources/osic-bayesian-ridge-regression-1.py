#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
df_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

print('Train shape: ', df_train.shape)
print('Number of unique customers in train: {}'.format(df_train['Patient'].nunique()))
print('Test shape:', df_test.shape)


# # Format data to meet submission requirement
# ## Train

# In[ ]:


df_base = df_train.drop_duplicates(subset='Patient', keep='first')
df_base = df_base[['Patient', 'Weeks', 'FVC', 
                   'Percent', 'Age']].rename(columns={'Weeks': 'base_week',
                                                      'Percent': 'base_percent',
                                                      'Age': 'base_age',
                                                      'FVC': 'base_FVC'})
df_base.head(3)


# In[ ]:


# Remove the first visit (base visit)
df_train['visit'] = 1
df_train['visit'] = df_train[['Patient', 'visit']].groupby('Patient').cumsum()
df_train = df_train.loc[df_train['visit'] > 1, :]


# In[ ]:


# Merge with base info
df_train = pd.merge(df_train,
                    df_base,
                    on='Patient',
                    how='left')
print(df_train.shape)
df_train.head(3)


# In[ ]:


df_train['weeks_passed'] = df_train['Weeks'] - df_train['base_week']
df_train = pd.get_dummies(df_train, columns=['Sex', 'SmokingStatus'])


# In[ ]:


df_train.head()


# ## Submission

# In[ ]:


sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
sub.head()


# In[ ]:


df_test = df_test.rename(columns={'Weeks': 'base_week', 
                                  'Percent': 'base_percent',
                                  'Age': 'base_age',
                                  'FVC': 'base_FVC'})
df_test = pd.merge(sub,
                   df_test,
                   on='Patient',
                   how='right')
df_test = pd.get_dummies(df_test, columns=['Sex', 'SmokingStatus'])
df_test['weeks_passed'] = df_test['Weeks'] - df_test['base_week']
df_test.head()


# In[ ]:


missing_columns = np.setdiff1d(df_train.drop(['Patient', 'FVC', 'Percent', 'Age', 'visit'], axis = 1).columns, df_test.columns)
if len(missing_columns) > 0:
    print('/!\ Missing columns in test: ', missing_columns)
    for col in missing_columns:
        df_test[col] = 0


# # Baseline

# In[ ]:


def OSIC_metric(y_true, y_pred, y_pred_std):
    delta = np.clip(abs(y_true - y_pred), 0, 1000)
    std_clipped = np.clip(y_pred_std, 70, np.inf)
    return np.mean(-(np.sqrt(2)*delta/std_clipped) - np.log(np.sqrt(2)*std_clipped))


# In[ ]:


from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge

class Model():
    def __init__(self, model=BayesianRidge(), n_splits=2):
        self.regressor = model
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=n_splits)
        self.train_cols = ['Weeks', 'base_week', 'base_FVC', 
                           'base_percent', 'base_age', 'weeks_passed', 'Sex_Female',
                           'Sex_Male', 'SmokingStatus_Currently smokes', 
                           'SmokingStatus_Ex-smoker', 'SmokingStatus_Never smoked']
    
    def fit(self, X, y):
        self.regressor.fit(X, y)
            
    def predict(self, X):
        pred = self.regressor.predict(X, return_std=True)        
        return pred
    
    def fit_predict_cv(self, df, df_test=pd.DataFrame()):
        
        scores = np.zeros((self.n_splits, ))
        oof = np.zeros((len(df), ))
        oof_std = np.zeros_like(oof)
        
        if len(df_test) > 0:
            pred_sub = np.zeros((len(df_test), self.n_splits))
            pred_sub_std = np.zeros_like(pred_sub)
        
        target = 'FVC'
        
        for i, (train_idx, val_idx) in enumerate(self.gkf.split(df, groups=df['Patient'])):
            X_train = df.loc[train_idx, self.train_cols]
            y_train = df.loc[train_idx, target]
            X_val = df.loc[val_idx, self.train_cols]
            y_val = df.loc[val_idx, target]
            
            self.fit(X_train, y_train)
            
            pred_train, pred_train_std = self.predict(X_train)
            pred_val, pred_val_std = self.predict(X_val)
            
            if len(df_test) > 0:
                pred_sub[:, i], pred_sub_std[:, i] = self.predict(df_test[self.train_cols])
            
            oof[val_idx] = pred_val
            oof_std[val_idx] = pred_val_std
            print('Train score: {0:.2f} | Test score: {1:.2f}'.format(OSIC_metric(y_train, pred_train, pred_train_std),
                                                                    OSIC_metric(y_val, pred_val, pred_val_std)))
        print('OOF score: {0:.4f}'.format(OSIC_metric(df[target], oof, oof_std)))
        res = dict()
        res['oof'] = oof
        res['oof_std'] = oof_std
        
        if len(df_test) > 0:
            res['pred_sub'] = pred_sub.mean(axis=1)
            res['pred_sub_std'] = pred_sub_std.mean(axis=1)
        
        return res


# In[ ]:


fvc_model = Model()


# In[ ]:


res = fvc_model.fit_predict_cv(df_train, df_test)


# In[ ]:


plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
sns.distplot(df_train['FVC'], label='Ground Truth')
sns.distplot(res['oof'], label='OOF')
plt.title('FVC Distributions')
plt.subplot(1, 2, 2)
sns.distplot(res['oof_std'])
plt.title('OOF Confidence Distribution')
plt.tight_layout()
plt.show()


# # Submission

# In[ ]:


df_test['FVC'] = res['pred_sub']
df_test['Confidence'] = res['pred_sub_std']

submission = sub[['Patient_Week']]
submission = pd.merge(submission,
                      df_test[['Patient_Week', 'FVC', 'Confidence']],
                      on='Patient_Week',
                      how='left')
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


# pip install jovian
#import jovian


# In[ ]:





# In[ ]:


jovian.commit(project='osic-bayesian')


# In[ ]:




