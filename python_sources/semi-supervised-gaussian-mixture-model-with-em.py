#!/usr/bin/env python
# coding: utf-8

# > ---
# # Theory

# In Gaussian Mixture model, we maximize likelihood function $P(X_{train}|\pi,\mu,\Sigma)$ about $\pi, \mu, \Sigma$ by using EM algorithm (here, $\pi$ means distribution parameter of label Y, $\mu$ and $\Sigma$ are set of mean vector and covariance matrix for each categories).
# 
# In this competiton, we have $X_{train}, Y_{tain}, X_{test}$. Gaussian Mixture model can treat $X_{train}, X_{test}$, and QDA (and other normal supervised model) can treat $X_{train}, Y_{train}$. Neither of them can treat $X_{train}, Y_{tain}, X_{test}$ at the same time.
# 
# Then I tried to modify EM algorithm for Gaussian Mixture as to treat these at the same time. Equations are slightly complicated, but result is not so difficult. Please see chapter 9 of PRML book for detail (I used notation Z for representation of label-variable instead of Y, because I deduct formulas based on PRML).

# $$
# \begin{align}
# P(X_{train},Y_{train},X_{test}|\pi,\mu,\Sigma)
# &= P(X_{train},Y_{train}|\pi,\mu,\Sigma)P(X_{test}|\pi,\mu,\Sigma) \\
# &= \Pi_{n=1}^{N_{train}}(\Pi_{k=1}^K \pi_k^{z_{nk}} N(x_n|\mu_k,\Sigma_k)^{z_{nk}})
# \Pi_{n=1}^{N_{test}}\Sigma_{k=1}^K \pi_k N(x_n|\mu_k,\Sigma_k)
# \end{align}
# $$
# and take log of this becomes below (log-likelihood).
# $$
# \log P = \Sigma_{n=1}^{N_{train}}\Sigma_{k=1}^K z_{nk}(\log \pi_k + \log N(x_n|\mu_k,\Sigma_k))
# +\Sigma_{n=1}^{N_{test}}\log\Sigma_{k=1}^K \pi_k N(x_n|\mu_k,\Sigma_k)
# $$

# take derivatives of this is
# $$
# \frac{\partial\log P}{\partial\mu_s}
# =\Sigma_{n=1}^{N_{train}}z_{ns}\Sigma^{-1}_s(x_n-\mu_s)
# +\Sigma_{n=1}^{N_{test}}\frac{\pi_s N(x_n|\mu_s,\Sigma_s)}{\Sigma_{k=1}^K\pi_kN(x_n|\mu_k,\Sigma_k)}
# \Sigma^{-1}_s(x_n-\mu_s)=0
# $$
# therefore, if we assume $\Sigma_s$ is non-singular and define as below,
# $$
# \gamma(z_{ns}):=\frac{\pi_s N(x_n|\mu_s,\Sigma_s)}{\Sigma_{k=1}^K\pi_k N(x_n|\mu_k,\Sigma_k)}
# $$
# we get
# $$
# \mu_s = \frac{\Sigma_{n=1}^{N_{train}}z_{ns}x_n+\Sigma_{n=1}^{N_{test}}\gamma(z_{ns}x_n)}
# {\Sigma_{n=1}^{N_{train}}z_{ns}+\Sigma_{n=1}^{N_{test}}\gamma(z_{ns})}.
# $$

# Let's compare this to normal Gaussian Mixture model.
# $$
# Normal Gaussian Mixture \\
# \mu_s = \frac{\Sigma_{n=1}^{N_{test}}\gamma(z_{ns}x_n)}
# {\Sigma_{n=1}^{N_{test}}\gamma(z_{ns})}.
# $$
# The meaning is very simple. We just use actual $z_{nk}$ instead of $\gamma(z_{ns})$ for taking weighted average step.

# Then I finally get
# $$
# \begin{align}
# \mu_s &=\frac{\Sigma_{n=1}^{N_{train}}z_{ns}x_n+\Sigma_{n=1}^{N_{test}}\gamma(z_{ns}x_n)}
# {\Sigma_{n=1}^{N_{train}}z_{ns}+\Sigma_{n=1}^{N_{test}}\gamma(z_{ns})} \\
# \Sigma_s &=\frac{\Sigma_{n=1}^{N_{train}}z_{ns}(x_n-\mu_s)(x_n-\mu_s)^T+\Sigma_{n=1}^{N_{test}}\gamma(z_{ns}(x_n-\mu_s)(x_n-\mu_s)^T)}
# {\Sigma_{n=1}^{N_{train}}z_{ns}+\Sigma_{n=1}^{N_{test}}\gamma(z_{ns})} \\
# \pi_s &=\frac{\Sigma_{n=1}^{N_{train}}z_{ns}+\Sigma_{n=1}^{N_{test}}\gamma(z_{ns})}
# {\Sigma_t(\Sigma_{n=1}^{N_{train}}z_{nt}+\Sigma_{n=1}^{N_{test}}\gamma(z_{nt}))}
# \end{align}
# $$
# (Sorry, I deduct other than $\mu_s$ intuitively, then they may be wrong).

# ---
# # Load Data

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

wcrms = df_train['wheezy-copper-turtle-magic'].unique()


# ---
# # Modeling

# In[ ]:


import numpy as np
from scipy import stats

class SSGaussianMixture(object):
    def __init__(self, n_features, n_categories):
        self.n_features = n_features
        self.n_categories = n_categories
        
        self.mus = np.array([np.random.randn(n_features)]*n_categories)
        self.sigmas = np.array([np.eye(n_features)]*n_categories)
        self.pis = np.array([1/n_categories]*n_categories)
        
        
    def fit(self, X_train, y_train, X_test, threshold=0.00001, max_iter=100):
        Z_train = np.eye(self.n_categories)[y_train] 
        
        for i in range(max_iter):
        # EM algorithm
            # M step
            Z_test = np.array([self.gamma(X_test, k) for k in range(self.n_categories)]).T
            Z_test /= Z_test.sum(axis=1, keepdims=True)
        
            # E step
            datas = [X_train, Z_train, X_test, Z_test]
            mus = np.array([self._est_mu(k, *datas) for k in range(self.n_categories)])
            sigmas = np.array([self._est_sigma(k, *datas) for k in range(self.n_categories)])
            pis = np.array([self._est_pi(k, *datas) for k in range(self.n_categories)])
            
            diff = max(np.max(np.abs(mus-self.mus)), 
                       np.max(np.abs(sigmas-self.sigmas)), 
                       np.max(np.abs(pis-self.pis)))
            #print(diff)
            self.mus = mus
            self.sigmas = sigmas
            self.pis = pis
            if diff<threshold:
                break
                
                
    def predict_proba(self, X):
        Z_pred = np.array([self.gamma(X, k) for k in range(self.n_categories)]).T
        Z_pred /= Z_pred.sum(axis=1, keepdims=True)
        return Z_pred


    def gamma(self, X, k):
        # X is input vectors, k is feature index
        return stats.multivariate_normal.pdf(X, mean=self.mus[k], cov=self.sigmas[k])
        
    def _est_mu(self, k, X_train, Z_train, X_test, Z_test):
        mu = (Z_train[:,k]@X_train + Z_test[:,k]@X_test).T /                  (Z_train[:,k].sum() + Z_test[:,k].sum())
        return mu
    
    def _est_sigma(self, k, X_train, Z_train, X_test, Z_test):
        cmp1 = (X_train-self.mus[k]).T@np.diag(Z_train[:,k])@(X_train-self.mus[k])
        cmp2 = (X_test-self.mus[k]).T@np.diag(Z_test[:,k])@(X_test-self.mus[k])
        sigma = (cmp1+cmp2) / (Z_train[:,k].sum() + Z_test[:k].sum())
        return sigma
        
    def _est_pi(self, k, X_train, Z_train, X_test, Z_test):
        pi = (Z_train[:,k].sum() + Z_test[:,k].sum()) /                  (Z_train.sum() + Z_test.sum())
        return pi
        


# In[ ]:


# Below is just a lapper object.

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

class BaseClassifier(object):
    def __init__(self):
        self.preprocess = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
 

    def fit(self, X_train, y_train, X_test, cv_qda=2, cv_meta=2):
        X_train_org = X_train
        self.preprocess_tune(np.vstack([X_train, X_test]))
        X_train = self.preprocess.transform(X_train)
        X_test = self.preprocess.transform(X_test)
        
        self.cgm = SSGaussianMixture(n_features=X_train.shape[1], n_categories=2)
        self.validation(X_train_org, y_train)
        self.cgm.fit(X_train, y_train, X_test)

    
    def predict(self, X):
        X = self.preprocess.transform(X)
        return self.cgm.predict_proba(X)[:,1]
    
    
    def preprocess_tune(self, X):
        self.preprocess.fit(X)
                
        
    def validation(self, X, y):
        X = self.preprocess.transform(X)
        kf = KFold(n_splits=3, shuffle=True)
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.cgm.fit(X_train, y_train, X_test)
            y_pred = self.cgm.predict_proba(X_test)[:,1]
            scores.append(roc_auc_score(y_test, y_pred))
        self.score = np.array(scores).mean()
        print('validation score = ', self.score)


# In[ ]:


df_train_sample = df_train[df_train['wheezy-copper-turtle-magic']==wcrms[3]]
X_train_sample = df_train_sample.drop(['id', 'target', 'wheezy-copper-turtle-magic'], axis=1).values
y_train_sample = df_train_sample['target'].values

df_test_sample = df_test[df_test['wheezy-copper-turtle-magic']==wcrms[3]]
X_test_sample = df_test_sample.drop(['id', 'wheezy-copper-turtle-magic'], axis=1).values


# In[ ]:


bc = BaseClassifier()
bc.fit(X_train_sample, y_train_sample, X_test_sample)


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
            self.clfs[id].fit(X_train, y_train, X_test)
            
        print('mean score = ', np.array([clf.score for clf in self.clfs.values()]).mean())


# In[ ]:


ce = ConsolEstimator(ids=wcrms)
ce.fit(df_train, df_test)


# ---
# # Submission

# In[ ]:


y_pred = ce.predict(df_test)


# In[ ]:


df_submission = pd.concat([df_test['id'], pd.Series(y_pred, name='target')], axis=1)
df_submission.to_csv('submission.csv', index=False)


# I tried hyper parameter tuning and bugging, but LB score didn't improve. 
# When I used BayesianGaussianMixture instead of GaussianMixture in stacking, my CV score improved, then if we can treat above model in Bayesian, score may improve.
# But for me, treating Gauss-Wishart distribution is hard (sklean implementation of GayesianGaussianMixture can be used as a refference).
