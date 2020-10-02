#!/usr/bin/env python
# coding: utf-8

# # Instant Gratification Solution (118th place, top 7%, bronze medal)

# ![](https://storage.googleapis.com/kaggle-media/competitions/general/Kerneler-white-desc2_transparent.png)

# ## 1. Dependencies and utility functions

# In[ ]:


# Dependencies
import subprocess
import re
import sys
import os
import glob
import warnings
import ctypes
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.covariance import GraphicalLasso
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Optimization code to make BLAS single-threaded

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


# Load the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# ## 2. Models

# ### 2.1. QDA with PL from GL+GMM

# In[ ]:


# Function that estimates mean and covariance using Graphical Lasso model
def get_mean_cov(x,y):
    model = GraphicalLasso(alpha=0.05)
    ones = (y==1).astype(bool)
    x2 = x[ones]
    model.fit(x2)
    p1 = model.precision_
    m1 = model.location_
    
    onesb = (y==0).astype(bool)
    x2b = x[onesb]
    model.fit(x2b)
    p2 = model.precision_
    m2 = model.location_
    
    ms = np.stack([m1, m2])
    ps = np.stack([p1, p2])
    return ms, ps


# In[ ]:


# Train GMM model

cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof = np.zeros(len(train))
preds = np.zeros(len(test))

with single_threaded(np):
    for i in tqdm(range(512)):
        train2 = train[train['wheezy-copper-turtle-magic']==i]
        test2 = test[test['wheezy-copper-turtle-magic']==i]
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3, train2['target']):

            ms, ps = get_mean_cov(train3[train_index, :],train2.loc[train_index]['target'].values)

            gm = GaussianMixture(n_components=2, init_params='kmeans', covariance_type='full',
                                 tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,
                                 means_init=ms, precisions_init=ps, random_state=1)
            gm.fit(np.concatenate([train3, test3], axis = 0))
            oof[idx1[test_index]] = gm.predict_proba(train3[test_index, :])[:, 0]
            preds[idx2] += gm.predict_proba(test3)[:, 0] / skf.n_splits

auc = roc_auc_score(train['target'], oof)
print('GMM CV: ',round(auc, 5))


# In[ ]:


# Collect numbers of useful features using Variance Threshold
# to use them in PCA as n_components

cat_dict = dict()

cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')

for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    cat_dict[i] = train3.shape[1]


# In[ ]:


# Train QDA model with PL from GMM

test['target'] = preds
oof1 = np.zeros(len(train))
preds1 = np.zeros(len(test))

with single_threaded(np):
    for k in tqdm(range(512)):
        train2 = train[train['wheezy-copper-turtle-magic']==k] 
        train2p = train2.copy(); idx1 = train2.index 
        test2 = test[test['wheezy-copper-turtle-magic']==k]
        
        # Using pseudolabels with confidence <= 0.15 or >= 0.85
        test2p = test2[(test2['target']<=0.15) | (test2['target']>=0.85)].copy()
        test2p.loc[test2p['target']>=0.5, 'target'] = 1
        test2p.loc[test2p['target']<0.5, 'target'] = 0 
        train2p = pd.concat([train2p, test2p], axis=0)
        train2p.reset_index(drop=True, inplace=True)

        sel = VarianceThreshold(threshold=1.5)
        sel.fit(train2p[cols])
        train3p = sel.transform(train2p[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3p, train2p['target']):
            test_index3 = test_index[test_index < len(train3)]
            clf = QuadraticDiscriminantAnalysis(reg_param=0.4)
            clf.fit(train3p[train_index, :],train2p.loc[train_index]['target'])
            oof1[idx1[test_index3]] += clf.predict_proba(train3[test_index3, :])[:, 1]

            preds1[test2.index] += clf.predict_proba(test3)[:, 1] / skf.n_splits
        
auc = roc_auc_score(train['target'], oof1)
print('Model 1 CV: ', round(auc, 5))


# ### 2.2. PCA+QDA with PL from GL+GMM

# In[ ]:


# Train PCA+QDA using previously computed predictions from GMM 
# and PCA n_components
test['target'] = preds
oof2 = np.zeros(len(train))
preds2 = np.zeros(len(test))

with single_threaded(np):
    for k in tqdm(range(512)):
        train2 = train[train['wheezy-copper-turtle-magic']==k] 
        train2p = train2.copy(); idx1 = train2.index 
        test2 = test[test['wheezy-copper-turtle-magic']==k]

        # Using pseudolabels with confidence <= 0.2 or >= 0.8
        test2p = test2[(test2['target']<=0.2) | (test2['target']>=0.8)].copy()
        test2p.loc[test2p['target']>=0.5, 'target'] = 1
        test2p.loc[test2p['target']<0.5, 'target'] = 0 
        train2p = pd.concat([train2p, test2p], axis=0)
        train2p.reset_index(drop=True, inplace=True)

        pca = PCA(n_components=cat_dict[k], random_state=1234)
        pca.fit(train2p[cols])
        train3p = pca.transform(train2p[cols])
        train3 = pca.transform(train2[cols])
        test3 = pca.transform(test2[cols])

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3p, train2p['target']):
            test_index3 = test_index[test_index<len(train3)]
            clf = QuadraticDiscriminantAnalysis(reg_param=0.4)
            clf.fit(train3p[train_index, :],train2p.loc[train_index]['target'])
            oof2[idx1[test_index3]] += clf.predict_proba(train3[test_index3, :])[:, 1]
            preds2[test2.index] += clf.predict_proba(test3)[:, 1] / skf.n_splits

auc = roc_auc_score(train['target'], oof2)
print('Model 2 CV: ', round(auc, 5))


# ### 2.3. Bagging QDA with PL from Model 1

# In[ ]:


# Train QDA with bagging and PL from Model 1

test['target'] = preds1 
oof3 = np.zeros(len(train))
preds3 = np.zeros(len(test))

with single_threaded(np):
    for k in tqdm(range(512)):
        train2 = train[train['wheezy-copper-turtle-magic']==k] 
        train2p = train2.copy(); idx1 = train2.index 
        test2 = test[test['wheezy-copper-turtle-magic']==k]

        # Using all test data as pseudolabels
        test2p = test2.copy()
        test2p.loc[test2p['target']>=0.5, 'target'] = 1
        test2p.loc[test2p['target']<0.5, 'target'] = 0 
        train2p = pd.concat([train2p, test2p], axis=0)
        train2p.reset_index(drop=True, inplace=True)

        sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     
        train3p = sel.transform(train2p[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3p, train2p['target']):
            test_index3 = test_index[test_index<len(train3)]
            clf = QuadraticDiscriminantAnalysis(reg_param=0.3)
            clf = BaggingClassifier(clf, n_estimators=200, random_state=333)
            clf.fit(train3p[train_index, :],train2p.loc[train_index]['target'])
            oof3[idx1[test_index3]] += clf.predict_proba(train3[test_index3, :])[:, 1]
            preds3[test2.index] += clf.predict_proba(test3)[:, 1] / skf.n_splits
        
auc = roc_auc_score(train['target'], oof3)
print('Model 3 CV: ', round(auc, 5))


# ### 2.4. QDA with iterative PL

# In[ ]:


# Train QDA with Grid Search on reg_param and iterative PL for 4 loops

oof4 = np.zeros(len(train))
preds4 = np.zeros(len(test))
params = [{'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}]
reg_params = np.zeros(512)

with single_threaded(np):
    for i in tqdm(range(512)):
        train2 = train[train['wheezy-copper-turtle-magic']==i]
        test2 = test[test['wheezy-copper-turtle-magic']==i]
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True, inplace=True)

        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
        pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
        data2 = pipe.fit_transform(data[cols])
        train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

        skf = StratifiedKFold(n_splits=11, random_state=42)
        for train_index, test_index in skf.split(train2, train2['target']):
            qda = QuadraticDiscriminantAnalysis()
            clf = GridSearchCV(qda, params, cv=4)
            clf.fit(train3[train_index, :],train2.loc[train_index]['target'])
            reg_params[i] = clf.best_params_['reg_param']
            oof4[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds4[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

    for itr in range(4):
        test['target'] = preds4
        # Using pseudolabels with confidence < 0.045 or > 0.955
        test.loc[test['target'] > 0.955, 'target'] = 1
        test.loc[test['target'] < 0.045, 'target'] = 0
        usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]
        new_train = pd.concat([train, usefull_test]).reset_index(drop=True)
        # Assign 0 or 1 to highly confident predictions
        new_train.loc[oof > 0.995, 'target'] = 1
        new_train.loc[oof < 0.005, 'target'] = 0
        oof4 = np.zeros(len(train))
        preds4 = np.zeros(len(test))
        for i in tqdm(range(512)):
            train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]
            test2 = test[test['wheezy-copper-turtle-magic']==i]
            idx1 = train[train['wheezy-copper-turtle-magic']==i].index
            idx2 = test2.index
            train2.reset_index(drop=True,inplace=True)

            data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
            pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
            data2 = pipe.fit_transform(data[cols])
            train3 = data2[:train2.shape[0]]
            test3 = data2[train2.shape[0]:]

            skf = StratifiedKFold(n_splits=11, random_state=42)
            for train_index, test_index in skf.split(train2, train2['target']):
                oof_test_index = [t for t in test_index if t < len(idx1)]
                clf = QuadraticDiscriminantAnalysis(reg_params[i])
                clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
                if len(oof_test_index) > 0:
                    oof4[idx1[oof_test_index]] = clf.predict_proba(train3[oof_test_index,:])[:,1]
                preds4[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

auc = roc_auc_score(train['target'], oof4)
print('Model 4 CV: ', round(auc, 5))


# ## 3. Final submission

# In[ ]:


# A weighted average of all the models
oof = 0.65*(0.25*oof1 + 0.15*oof2 + 0.6*oof3) + 0.35*oof4
preds = 0.65*(0.25*preds1 + 0.15*preds2 + 0.6*preds3) + 0.35*preds4
auc = roc_auc_score(train['target'], oof)
print('Final submission CV: ', round(auc, 6))


# In[ ]:




