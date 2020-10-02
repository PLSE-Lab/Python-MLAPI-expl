#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.special import comb, logsumexp, expit
from scipy.stats import rankdata
from tqdm import tqdm_notebook
np.random.seed(1111)
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/instant-gratification/train.csv')
test = pd.read_csv('../input/instant-gratification/test.csv')
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


gm_list = pickle.load(open('../input/models-v5/gm_models_v5.pkl', 'rb'))


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
            return logsumexp(weighted_log_prob[:, 1::2], axis=1) - logsumexp(weighted_log_prob[:, 0::2], axis=1)
        


# In[ ]:


class MyBGM(BayesianGaussianMixture):

    
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10, init_clusters=None):
        super().__init__(
                n_components=n_components, covariance_type=covariance_type, tol=tol,
                 reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                 weight_concentration_prior_type=weight_concentration_prior_type,
                 weight_concentration_prior=weight_concentration_prior,
                 mean_precision_prior=mean_precision_prior, mean_prior=mean_prior,
                 degrees_of_freedom_prior=degrees_of_freedom_prior, covariance_prior=covariance_prior,
                 random_state=random_state, warm_start=warm_start, verbose=verbose,
                 verbose_interval=verbose_interval)
        self.init_clusters_ = np.asarray(init_clusters).astype('int')
        
        
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
        elif self.init_params == 'proba':
            resp = self.init_proba_.copy()
            resp[np.arange(self.init_clusters_.shape[0]), self.init_clusters_] = 1
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize(X, resp)


# In[ ]:


reg_covar=1.986808882
tol=0.012220889
#tol=0.05
gm_list_save = []
if True:
    oof_preds = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])
    n_random = 5
    for RANDOM_STATE in tqdm_notebook(range(n_random)):
        for i in tqdm_notebook(range(512)):
            train2 = train[train['wheezy-copper-turtle-magic']==i]
            test2 = test[test['wheezy-copper-turtle-magic']==i]
            idx1 = train2.index
            idx2 = test2.index
            train2.reset_index(drop=True,inplace=True)
            sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
            train3 = sel.transform(train2[cols])
            test3 = sel.transform(test2[cols])

            train_cls = gm_list[i].predict(train3)
            pred_org = pd.crosstab(train_cls, train2['target']).values
            pred = pred_org[:, 1] / pred_org.sum(axis=1)

            skf = StratifiedKFold(n_splits=11, random_state=42 + RANDOM_STATE, shuffle=True)

            for train_index, test_index in skf.split(train3, train_cls):

                # concat train and test
                train_test3 = np.concatenate([train3[train_index], test3], axis=0)
                train_test_cls = gm_list[i].predict(train_test3)

                gm_all = MyGM(
                    n_components=6, init_params='clusters', init_clusters=train_test_cls, 
                    weights_init=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],  
                    reg_covar=reg_covar, tol=tol
                )
                _ = gm_all.fit(train_test3)
                gm_list_save.append(gm_all)
                if False:
                    oof_pred6 = gm_all.predict_proba(train3[test_index])
                    oof_preds[idx1[test_index]] += oof_pred6[:, pred>0.5].sum(axis=1)
                    test_pred6 = gm_all.predict_proba(test3)
                    test_preds[idx2] += test_pred6[:, pred>0.5].sum(axis=1)
                else:
                    oof_pred6 = gm_all._estimate_weighted_log_prob(train3[test_index])
                    oof_preds[idx1[test_index]] += logsumexp(oof_pred6[:, pred>0.5], axis=1) - logsumexp(oof_pred6[:, pred<=0.5], axis=1)
                    test_pred6 = gm_all._estimate_weighted_log_prob(test3)
                    test_preds[idx2] += logsumexp(test_pred6[:, pred>0.5], axis=1) - logsumexp(test_pred6[:, pred<=0.5], axis=1)
                
        print(roc_auc_score(train["target"], oof_preds))


# In[ ]:


_ = plt.hist(test_preds)


# In[ ]:


_ = plt.hist(oof_preds)


# In[ ]:


roc_auc_score(train["target"], oof_preds)


# In[ ]:


sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')
sub['target'] = test_preds
sub.to_csv('submission.csv',index=False)


# In[ ]:




