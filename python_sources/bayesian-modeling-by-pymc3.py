#!/usr/bin/env python
# coding: utf-8

# # Bayesian modeling by PyMC3
# This Kernel is not practilcal because of run-time is too long. Then cannot be used for submission.

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


print('train.shape={}'.format(df_train.shape), 'test_shape={}'.format(df_test.shape))


# ---
# ## Modeling
# It is based on the below official documents.  
# https://docs.pymc.io/notebooks/gaussian_mixture_model.html

# In[ ]:


wcrms = df_train['wheezy-copper-turtle-magic'].unique()


# In[ ]:


# I assumed covariance matrix has only diagonal factors.
# I tried including off-diagonal factors, but failed.


import pymc3 as pm
import theano.tensor as tt


class BayesianClassifier(object):
    def __init__(self, n_categories=2):
        self.n_categories = n_categories
        self.model = pm.Model()
    
    
    def predict(self, X=None):
        # input X is just a dummy for keep API the same.
        y_pred = self.trace['z'][:,self.n_trains:].mean(axis=0)
        return y_pred
    
    
    def fit(self, X_train, y_train, X_test):
        # initial value for mu and tau can be improved by GaussianMixture of sklearn.
        X = np.concatenate([X_train, X_test], axis=0)
        self.n_features = X.shape[1]
        self.n_trains = X_train.shape[0]
        default_mu = 0.1 * np.random.randn(self.n_categories, self.n_features)
        with self.model:
            p = pm.Dirichlet('p', a=np.array([1.]*self.n_categories), shape=self.n_categories)
            z = pm.Categorical('z', p=p, shape=len(X))
            p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p)<0.1, -np.inf,0))

            mu = pm.Normal('mu', mu=default_mu, tau=1/1**2, shape=(self.n_categories, self.n_features))    
            sd = pm.Uniform('sd', lower=0.01, upper=5, shape=(self.n_categories, self.n_features))
            
            z_obs = pm.Normal('z_obs', mu=z[:len(y_train)], tau=100, observed=y_train.ravel())
            X_obs = pm.Normal('X_obs', mu=mu[z,:], tau=1/sd[z,:], observed=X)
            
            self.trace = pm.sample(3000)


# In[ ]:


class BaseClassifier(object):
    # just a lapper object.
    def __init__(self, n_folds=10):
        self.clf = BayesianClassifier(2)
        
    
    def fit(self, X_train, y_train, X_test, n_folds=5):
        X_train = self._apply_feature_mask(X_train)
        X_test = self._apply_feature_mask(X_test)
        self.clf.fit(X_train, y_train, X_test)
        
    
    def predict(self, X=None):
        return self.clf.predict(X)
    
    
    def fit_feature_mask(self, X, threshold=1-45/256):
        stds = X.std(axis=0)
        split = np.quantile(stds, threshold)
        self.feature_mask = (split<=stds)
        
        
    def _apply_feature_mask(self, X):
        return X[:, self.feature_mask]
    
    


# ---
# ## Evaluation
# Let's evaluate Bayesian classifier to QDA.

# In[ ]:


from sklearn.model_selection import train_test_split
sample_df = df_train[df_train['wheezy-copper-turtle-magic']==wcrms[3]]
X_train_sample = sample_df.drop(['id', 'target', 'wheezy-copper-turtle-magic'], axis=1).values
y_train_sample = sample_df['target'].values

X_train_sample, X_val_sample, y_train_sample, y_val_sample =     train_test_split(X_train_sample, y_train_sample)


# In[ ]:


bc = BaseClassifier()
bc.fit_feature_mask(X_train_sample)
# It may take much time starting sampling...
bc.fit(X_train_sample, y_train_sample, X_val_sample)


# In[ ]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

qda = QuadraticDiscriminantAnalysis(2, reg_param=0.8)
qda.fit(X_train_sample, y_train_sample)
print('AUC of QDA is : ', roc_auc_score(y_val_sample, qda.predict_proba(X_val_sample)[:,1]), '.')
print('AUC of Baysean is : ', roc_auc_score(y_val_sample, bc.predict(X_val_sample)), '.')


# ---
# Below code are for final submission, but it cannot end in the time.

# In[ ]:


class ConsolEstimator(object):
    def __init__(self, ids):
        self.clfs = {}
        self.id_column = 'wheezy-copper-turtle-magic'
        self.ids = ids
        
        
    def predict(self, df_X):
        y_pred = np.zeros(shape=(len(df_X)))
        for id in df_X[self.id_column].unique():
            id_rows = (df_X[self.id_column]==id)
            X = df_X.drop(['id', self.id_column], axis=1).values[id_rows]
            y_pred[id_rows] = self.clfs[id].predict(X)
        return y_pred
            
        
    def fit(self, df_train, df_test):
        for i, id in enumerate(self.ids):
            print(i, 'th training...')
            df_train_id = df_train[df_train[self.id_column]==id]
            df_test_id = df_test[df_test[self.id_column]==id]
            if len(df_train_id)==0 or len(df_test_id)==0:
                continue
            
            X_train = df_train_id.drop(['id', 'target', self.id_column], axis=1).values
            y_train = df_train_id['target'].values
            X_test = df_test_id.drop(['id', self.id_column], axis=1).values
            
            self.clfs[id] = BaseClassifier()
            self.clfs[id].fit_feature_mask(np.vstack([X_train, X_test]))
            self.clfs[id].fit(X_train, y_train, X_test)
            


# In[ ]:


# testing code
#df_train_sample = df_train[df_train['wheezy-copper-turtle-magic'].isin(wcrms[:10])]
#df_test_sample = df_test[df_test['wheezy-copper-turtle-magic'].isin(wcrms[:10])]


# In[ ]:


#ce = ConsolEstimator(ids=wcrms)
#ce.fit(df_train_sample, df_test_sample)

# final run.
#ce.fit(df_train, df_test)


# ---
# ## Submission

# In[ ]:


#y_pred = ce.predict(df_test)


# In[ ]:


#df_submission = pd.concat([df_test['id'], pd.Series(y_pred, name='target')], axis=1)
#df_submission.to_csv('submission.csv', index=False)

