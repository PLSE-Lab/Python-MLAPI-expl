#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess
import re
import sys
import os
import glob
import warnings
import ctypes

_MKL_ = 'mkl'
_OPENBLAS_ = 'openblas'


class BLAS:
    def __init__(self, cdll, kind):
        if kind not in (_MKL_, _OPENBLAS_):
            raise ValueError(f'kind must be {MKL} or {OPENBLAS}, got {kind} instead.')
        
        self.kind = kind
        self.cdll = cdll
        
        if kind == _MKL_:
            self.get_n_threads = cdll.MKL_Get_Max_Threads
            self.set_n_threads = cdll.MKL_Set_Num_Threads
        else:
            self.get_n_threads = cdll.openblas_get_num_threads
            self.set_n_threads = cdll.openblas_set_num_threads
            

def get_blas(numpy_module):
    LDD = 'ldd'
    LDD_PATTERN = r'^\t(?P<lib>.*{}.*) => (?P<path>.*) \(0x.*$'

    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, '_multiarray_umath.*so'))[0]
    ldd_result = subprocess.run(
        args=[LDD, MULTIARRAY_PATH], 
        check=True,
        stdout=subprocess.PIPE, 
        universal_newlines=True
    )

    output = ldd_result.stdout

    if _MKL_ in output:
        kind = _MKL_
    elif _OPENBLAS_ in output:
        kind = _OPENBLAS_
    else:
        return

    pattern = LDD_PATTERN.format(kind)
    match = re.search(pattern, output, flags=re.MULTILINE)

    if match:
        lib = ctypes.CDLL(match.groupdict()['path'])
        return BLAS(lib, kind)
    

class single_threaded:
    def __init__(self, numpy_module=None):
        if numpy_module is not None:
            self.blas = get_blas(numpy_module)
        else:
            import numpy
            self.blas = get_blas(numpy)

    def __enter__(self):
        if self.blas is not None:
            self.old_n_threads = self.blas.get_n_threads()
            self.blas.set_n_threads(1)
        else:
            warnings.warn(
                'No MKL/OpenBLAS found, assuming NumPy is single-threaded.'
            )

    def __exit__(self, *args):
        if self.blas is not None:
            self.blas.set_n_threads(self.old_n_threads)
            if self.blas.get_n_threads() != self.old_n_threads:
                message = (
                    f'Failed to reset {self.blas.kind} '
                    f'to {self.old_n_threads} threads (previous value).'
                )
                raise RuntimeError(message)
    
    def __call__(self, func):
        def _func(*args, **kwargs):
            self.__enter__()
            func_result = func(*args, **kwargs)
            self.__exit__()
            return func_result
        return _func


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

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# My GM, reusing Hoxosh likelihood, with some modification.
# 

# In[ ]:


class MyGM(GaussianMixture):
    
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=0, warm_start=False,
                 verbose=0, verbose_interval=10, init_clusters=None, y=None, contamination=0.025, init_proba=None):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.init_clusters_ = np.asarray(init_clusters).astype('int')
        self.y_ = y
        self.contamination_ = contamination
        self.init_proba_ = init_proba
        
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
     
    def estimate_log_ratio(self, X):
            weighted_log_prob = self._estimate_weighted_log_prob(X)
            
            return logsumexp(weighted_log_prob[:, 1::2], axis=1) - logsumexp(weighted_log_prob[:, 0::2], axis=1)
        
    # override estimate step
    # for this competition
    def _estimate_log_prob(self, X):
        base_pred = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)
        
        if self.y_ is None:
            new_pred = base_pred
        else:
            new_pred = base_pred.copy()
            for i in range(self.init_clusters_.shape[0]):
                yi = self.init_clusters_[i]
                for j in range(base_pred.shape[1]):
                    if (j - yi) % 2 == 0:
                        new_pred[i, j] += np.log(1 - self.contamination_)
                    else:
                        new_pred[i, j] += np.log(self.contamination_)
        return new_pred
    


# In[ ]:


if test.shape[0] != 131073:
    with single_threaded(np):
        oof_preds = np.zeros(train.shape[0])
        test_preds = np.zeros(test.shape[0])
        num_models = 512
        n_splits = 7
        contamination = 0.025
        n_cluster = 3
        n_components = 2 * n_cluster
        reg_covar=1.986808882
        tol=0.012220889
        n_random = 4
        for RANDOM_STATE in tqdm_notebook(range(n_random)):
            for i in tqdm_notebook(range(num_models)):

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
                n_features = train3.shape[1]

                skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

                for train_index, test_index in skf.split(train3, train2['target']):

                    # initialize with train, 3 clusters each
                    train_x_0 = train3[train_index][train2['target'].values[train_index] == 0]
                    train_x_1 = train3[train_index][train2['target'].values[train_index] == 1]

                    gm0 = GaussianMixture(n_components=n_cluster, weights_init=[1/n_cluster] * n_cluster,
                                          n_init=5, random_state=RANDOM_STATE)
                    _ = gm0.fit(train_x_0)
                    cls0 = gm0.predict(train_x_0)
                    ms0 = gm0.means_
                    ps0 = gm0.precisions_
                    gm1 = GaussianMixture(n_components=n_cluster, #weights_init=[1/n_cluster] * n_cluster,
                                          n_init=5, reg_covar=reg_covar, tol=tol, random_state=RANDOM_STATE)
                    _ = gm1.fit(train_x_1)
                    cls1 = gm1.predict(train_x_1)
                    ms1 = gm1.means_
                    ps1 = gm1.precisions_

                    train_cls = np.ones(train3[train_index].shape[0]).astype(np.int)
                    train_cls[train2['target'].values[train_index] == 0] = cls0 * 2
                    train_cls[train2['target'].values[train_index] == 1] = cls1 * 2 + 1

                    train_ms = np.zeros((n_components, n_features))
                    train_ms[0::2] = ms0
                    train_ms[1::2] = ms1

                    train_ps = np.zeros((n_components, n_features, n_features))
                    train_ps[0::2] = ps0
                    train_ps[1::2] = ps1

                    # concat train and test
                    train_test3 = np.concatenate([train3[train_index], test3], axis=0)

                    gm_all = MyGM(
                        n_components=n_components, init_params='clusters', init_clusters=train_cls, 
                        means_init=train_ms, precisions_init=train_ps,
                        weights_init=[1/n_components] * n_components, reg_covar=reg_covar, tol=tol, contamination=contamination, y=None,
                    )
                    _ = gm_all.fit(train_test3)
                    gm_all.y_ = None

                    if True:
                        oof_pred6 = gm_all.estimate_log_ratio(train3[test_index])
                        test_pred6 = gm_all.estimate_log_ratio(test3) 
                    else:
                        oof_pred6 = gm_all.predict_proba(train3[test_index])
                        oof_pred6 = oof_pred6[:, 1::2].sum(axis=1) / oof_pred6.sum(axis=1) 
                        test_pred6 = gm_all.predict_proba(test3)
                        test_pred6 = test_pred6[:, 1::2].sum(axis=1) / test_pred6.sum(axis=1)
                    oof_preds[idx1[test_index]] += oof_pred6
                    test_preds[idx2] += test_pred6

        oof_preds /= n_random
        test_preds /= skf.n_splits * n_random
        train_used = (train['wheezy-copper-turtle-magic'] < num_models)
        auc = roc_auc_score(train.loc[train_used, 'target'], oof_preds[train_used])
        print(f'AUC: {auc:.5}')


# In[ ]:


if test.shape[0] != 131073:
    train_used = (train['wheezy-copper-turtle-magic'] < num_models)
    auc = roc_auc_score(train.loc[train_used, 'target'], expit(oof_preds[train_used]))
    print(f'AUC: {auc:.5}')


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
if sub.shape[0] != 131073:
    sub['target'] = expit(test_preds)
sub.to_csv('submission.csv',index=False)


# In[ ]:


plt.hist((oof_preds), bins=128, log=True)
plt.title('Final oof predictions')
plt.show()


# In[ ]:


plt.hist(test_preds, bins=128, log=True)
plt.title('Final oof predictions')
plt.show()


# In[ ]:




