#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _estimate_log_gaussian_prob
from scipy.special import comb, logsumexp, expit
from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# My GM, modified CPMP's

# In[ ]:


class MyGM(GaussianMixture):
    
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10, init_clusters=None, y=None):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.init_clusters_ = np.asarray(init_clusters).astype('int')
        self.y = y
        
    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.
        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == 'clusters':
            resp = np.zeros((n_samples, self.n_components))
            resp[np.arange(self.init_clusters_.shape[0]), self.init_clusters_] = 1
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize(X, resp)
     
    def estimate_log_ratio(self, X):
            weighted_log_prob = self._estimate_weighted_log_prob(X)            
            return weighted_log_prob[:, 1] - weighted_log_prob[:, 0]
        
    # override estimate step
    # for this competition
    def _estimate_log_prob(self, X):
        base_pred = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)
        new_pred = base_pred
        
        if self.y is None:
            pass
        else:
            for i in range(base_pred.shape[0]):
                yi = self.y[i]
                if yi not in [0, 1]:
                    pass
                else:
                    for j in range(base_pred.shape[1]):
                        if (j - yi) % 2 == 0:
                            new_pred[i, j] += np.log(0.975)
                        else:
                            new_pred[i, j] += np.log(0.025)
        return new_pred
    


# In[ ]:


oof_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])


# In[ ]:



for i in tqdm_notebook(range(512)):
    
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index
    idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    
    for train_index, test_index in skf.split(train3, train2['target']):

        # initialize with train, 3 clusters each
        train_x_0 = train3[train_index][train2['target'].values[train_index] == 0]
        train_x_1 = train3[train_index][train2['target'].values[train_index] == 1]

        gm0 = GaussianMixture(n_components=3)
        _ = gm0.fit(train_x_0)
        cls0 = gm0.predict(train_x_0)
        gm1 = GaussianMixture(n_components=3)
        _ = gm1.fit(train_x_1)
        cls1 = gm1.predict(train_x_1)

        # concat train and test
        train_test3 = np.concatenate([train3[train_index], test3], axis=0)

        train_cls = np.ones(train3[train_index].shape[0]).astype(np.int)
        train_cls[train2['target'].values[train_index] == 0] = cls0 * 2
        train_cls[train2['target'].values[train_index] == 1] = cls1 * 2 + 1

        gm_train = MyGM(
            n_components=6, init_params='clusters', init_clusters=train_cls, 
            weights_init=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],  reg_covar=1, tol=2, y=train2['target'].values[train_index]
        )
        _ = gm_train.fit(train3[train_index])
        gm_train.y = None
        train_test_cls = gm_train.predict(train_test3)

        gm_all = MyGM(
            n_components=6, init_params='clusters', init_clusters=train_test_cls, 
            weights_init=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],  reg_covar=1, tol=2, 
            y=np.concatenate([train2['target'].values[train_index], 2*np.ones(test3.shape[0]).astype(np.int)], axis=0)
        )
        gm_all.y = None
        _ = gm_all.fit(train_test3)
        
        oof_pred6 = gm_all.predict_proba(train3[test_index])
        oof_preds[idx1[test_index]] = oof_pred6[:, 1::2].sum(axis=1) / oof_pred6.sum(axis=1)
        test_pred6 = gm_all.predict_proba(test3)
        test_preds[idx2] += test_pred6[:, 1::2].sum(axis=1) / test_pred6.sum(axis=1)


# In[ ]:


roc_auc_score(train["target"], oof_preds)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = test_preds
sub.to_csv('submission.csv',index=False)

