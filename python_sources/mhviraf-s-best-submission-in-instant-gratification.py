#!/usr/bin/env python
# coding: utf-8

# This is the final version of the kernel I used for this competition. I am writing this note before the deadline so I am not sure if this is the exact same one I am going to be using for submission or not but regardless of that, my final submission is gonna be this kernel maybe with different number of iterations or seed number.
# 
# the only unrevealed key to this competition was to cluster each class before modeling it. I explained picturially why it works at https://www.kaggle.com/mhviraf/instant-gratification-lb-score-0-974. Everything else was revealed by the many genious and talented people in this competition. They were not only smart but kind and generous enough to share their findings. I am not gonna name them one by one here because we all know who they are.
# 
# I experimented so many different things for this competition including:
# * **different number of models to blend**: I believed the more number of models I blend together the more robust answer I would get so I tried to maximize this parameter and have the maximum number of models I could have in 1 run which ended up being 42 models of the same type.
# * **different number of folds per model**: this item was another important factor because it had a trade-off in it. the more folds I used, the more robust predictions. I achieved robustness by increasing number of models I used though, so I used this parameter for regularization. I chose 2 folds in the end so that despite we have only 512 samples per each `wheezy-copper-turtle-magic` I only used half of them to train a model and the other half to validate my model. I noticed if my model can have AUC 0.9748+ on only 50% of the data and generalize well on both the training set and public leader board, why not? let's continue with this.
# * **different clustering algorithms**: because I knew the data was multivariate gaussian distribution around the nodes of a hypercube (refer to `make_classification` documentation on sklearn and its source code on github) I assumed the best way to cluster them would be by using `mixture.GaussianMixture` and I ended up using it too. However, I though a lot about different clustering algorithms and studied the document at https://scikit-learn.org/stable/modules/clustering.html carefully and experimented with other clustering algorithms I believed might be useful. 
# * **different number of clusters**: I used the elbow rule and tsne and other algorithms to figure this one out but I couldn't verify what is the exact number of cluster per class they used in data generation. Nonetheless, I ran an experiment on my own synthetic data and by analyzing 1000 experiments I figured whether they are 2 or 3 clusters in the data itself if I lump data into 3 clusters I would get better validation and test AUCs so I continued with 3 clusters per class.
# * **different classifier algorithms**: It was kinda obvious that GMM is the way to go (Thanks to @christofhenkel, he was very underrepresented in this competition despite his great contributions. You have my respect). Nonetheless, I tried two different algorithms as well.
# * **different scalers**: First of all I figured I need to add scaler to ensure non-ill-conditioned matrices for covariance calculations. I tried two different scalers, results were not that much different but I continued with standardscaler again because features were linear combination of standard normal distributions. 
# * **different regularizations**: other than using 2 folds for training I used regularization parameters in both of the clustering and classifier algorithms. In the end I decided to go with their default values since training on 50% of the data was enough regularization. 
# * **model by model hyperparameter tuning and regularization**: I tried these options as well but they didn't work out well for me. 
# * etc. etc.

# In[ ]:


# from https://www.kaggle.com/sggpls/singlethreaded-instantgratification to speed things up
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


import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook as tqdm
from sklearn.cluster import KMeans, Birch, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso, EmpiricalCovariance, OAS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/instant-gratification/train.csv').sort_values(by='wheezy-copper-turtle-magic')
train_gmm_ids = train.index
train = train.reset_index(drop=True)
test = pd.read_csv('../input/instant-gratification/test.csv')
reg_params = pd.read_csv('../input/ig-best-reg-params/best_reg_.csv')


# In[ ]:



ITERS = 42
#np.random.seed(321321)
NFOLD = 2

RANDOM_SEED = 4
MODIFY_MEANS = False
GMM_N_INIT = 1
GMM_reg_ = 1e-6
GMM_INIT_PARAMS = 'random' # 'random' 'kmeans'
RUN_FROM = 0
RUN_TO = 512 # 512

CLUSTERING_ALGORITHM = 'gmm' # 'kmeans', 'gmm', 'birch', 'agg'
## if CLUSTERING_ALGORITHM = 'gmm', ignore otherwise
CLUSTERING_GMM_INIT_PARAMS = 'random' # 'random', 'kmeans'
CLUSTER_GMM_N_INIT = 1 # if any
N_CLUSTERS = 3

CLASSIFIER_ALGORITHM = 'gmm' # 'gmm', bgm', 'qda'

COVARIANCE_ALGORITHM = 'graphicallasso' # 'oas', 'graphicallasso', 'graphicallassoCV'

Scaler = 'standard' # 'standard', 'robust'


# In[ ]:


### CLUSTERING ALGORITHM  ********************************************************************************
if CLUSTERING_ALGORITHM == 'kmeans':
    knn_clf = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_jobs=-1)
    
elif CLUSTERING_ALGORITHM == 'gmm':
    knn_clf = GaussianMixture(n_components=N_CLUSTERS, init_params=CLUSTERING_GMM_INIT_PARAMS,
                              covariance_type='full',
                              n_init=CLUSTER_GMM_N_INIT, 
                              random_state=RANDOM_SEED, reg_covar=0.1)
elif CLUSTERING_ALGORITHM == 'birch':
    knn_clf = Birch(n_clusters=N_CLUSTERS, threshold=0.6, branching_factor=60)
elif CLUSTERING_ALGORITHM == 'agg':
    knn_clf = AgglomerativeClustering(n_clusters=2)


# In[ ]:


def get_mean_cov(X):
    if COVARIANCE_ALGORITHM == 'graphicallassoCV':
        model = GraphicalLassoCV(n_jobs=-1)
    elif COVARIANCE_ALGORITHM == 'graphicallasso':
        model = GraphicalLasso()
    elif COVARIANCE_ALGORITHM == 'oas':
        model = OAS(assume_centered=False)
    
    ms = []
    ps = []
    for xi in X:
        model.fit(xi)
        ms.append(model.location_)
        ps.append(model.precision_)
    return np.array(ms), np.array(ps)


# # make predictions

# In[ ]:


get_ipython().run_cell_magic('time', '', "RANDOM_SEED = 4\nfinal_preds = np.zeros(len(test))\nfinal_preds_ranked = np.zeros(len(test))\nfinal_auc = np.zeros(len(train))\n\nwith single_threaded(np):\n    for __ in range(ITERS):\n        dudes = range(RUN_FROM, RUN_TO)\n        # INITIALIZE VARIABLES\n        cols = [c for c in train.columns if c not in ['id', 'target']]\n        cols.remove('wheezy-copper-turtle-magic')\n        preds = np.zeros(len(test))\n        overal_oof = np.zeros(len(train))\n        overal_oof_y = np.zeros(len(train))\n        aucs = []\n        gmm_converged = []\n\n        min_idx, max_idx = 10000000, 0\n        # BUILD 512 SEPARATE MODELS\n        for i in dudes:\n            # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I\n            train2 = train[train['wheezy-copper-turtle-magic']==i]\n            test2 = test[test['wheezy-copper-turtle-magic']==i]\n            idx1 = train2.index; idx2 = test2.index\n            min_idx = min(min_idx, min(idx1))\n            max_idx = max(max_idx, max(idx1))\n            train2.reset_index(drop=True,inplace=True)\n\n            # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)\n            sel = VarianceThreshold(threshold=1.5).fit(train2[cols])\n            train3 = sel.transform(train2[cols])\n            test3 = sel.transform(test2[cols])\n\n            # STANDARDIZE\n            if Scaler == 'standard':\n                sclr = StandardScaler()\n            elif Scaler == 'robust':\n                sclr = RobustScaler()\n\n            train_test = sclr.fit_transform(np.vstack([train3, test3]))\n            train3 = train_test[:len(train3)]\n            test3 = train_test[len(train3):]\n\n            # CLUSTERING \n            ## FIND CLUSTERS IN CHUNKS WITH TARGET = 1\n            train3_pos = train3[train2['target']==1]\n            cluster_num_pos = knn_clf.fit_predict(train3_pos)\n            train3_pos_1 = train3_pos[cluster_num_pos==0]\n            train3_pos_2 = train3_pos[cluster_num_pos==1]\n            train3_pos_3 = train3_pos[cluster_num_pos==2]\n            #print(train3_pos.shape, train3_pos_1.shape, train3_pos_2.shape, train3_pos_3.shape)\n\n            ## FIND CLUSTERS IN CHUNKS WITH TARGET = 0\n            train3_neg = train3[train2['target']==0]\n            cluster_num_neg = knn_clf.fit_predict(train3_neg)\n            train3_neg_1 = train3_neg[cluster_num_neg==0]\n            train3_neg_2 = train3_neg[cluster_num_neg==1]\n            train3_neg_3 = train3_neg[cluster_num_neg==2]\n            #print(train3_neg.shape, train3_neg_1.shape, train3_neg_2.shape, train3_neg_3.shape)\n\n            four_class_train_X = np.vstack([train3_pos_1, train3_pos_2, train3_pos_3, train3_neg_1, train3_neg_2, train3_neg_3])\n            four_class_train_y = np.concatenate([np.zeros(len(train3_pos_1)),\n                                                 np.ones(len(train3_pos_2))*1,\n                                                 np.ones(len(train3_pos_3))*2,\n                                                 np.ones(len(train3_neg_1))*3,\n                                                 np.ones(len(train3_neg_2))*4,\n                                                np.ones(len(train3_neg_3))*5]).astype('int')\n            #print(four_class_train_X.shape, four_class_train_y.shape)\n\n            ys = np.concatenate([np.ones(train3_pos.shape[0]), np.zeros(train3_neg.shape[0])])\n            #print(train3_pos_1.shape, train3_pos_2.shape, train3_neg_1.shape, train3_neg_2.shape)\n            ms, ps = get_mean_cov([train3_pos_1, train3_pos_2, train3_pos_3, train3_neg_1, train3_neg_2, train3_neg_3])\n            #print(ms.shape, ps.shape)\n            overal_oof_y[idx1] = ys\n\n            oof = np.zeros(len(four_class_train_X))    \n            skf = StratifiedKFold(n_splits=NFOLD, random_state=RANDOM_SEED, shuffle=True)\n            for train_index, test_index in skf.split(four_class_train_X, four_class_train_y):\n                #train4 = train3[train_index]\n\n        #         if reg_params.loc[i, 'base_auc'] > reg_params.loc[i, 'best auc']:\n        #             reg_ = 1e-6\n        #         else: \n        #             reg_ = reg_params.loc[i, 'reg_param']\n                gm = GaussianMixture(n_components=6, init_params=GMM_INIT_PARAMS, \n                                 n_init=GMM_N_INIT, random_state=RANDOM_SEED,\n                                 means_init=ms, precisions_init=ps,\n                                 tol=0.00001, max_iter=5000, reg_covar=GMM_reg_)\n                gm.fit(np.vstack([four_class_train_X, test3]))\n                gmm_converged.append(gm.converged_)\n\n                oof_preds_initial = gm.predict_proba(four_class_train_X[test_index])\n                oof_preds = (oof_preds_initial[:,0:N_CLUSTERS].sum(axis=1)) / NFOLD\n\n                oof[test_index] += oof_preds\n                overal_oof[idx1[test_index]] += oof_preds\n\n                test_preds_initial = gm.predict_proba(test3)\n                preds[idx2] += (test_preds_initial[:,0:N_CLUSTERS].sum(axis=1)) / NFOLD\n\n            auc = roc_auc_score(ys, oof)\n            aucs.append(auc)\n\n        # PRINT CV AUC\n        print('split aucs: average=', np.mean(aucs), ' & std=', np.std(aucs))\n        print('overall AUC: ', roc_auc_score(overal_oof_y[min_idx:max_idx], overal_oof[min_idx:max_idx]))\n        #print('classification GMM not converged in: ', np.where(gmm_converged == True))\n\n        preds_gmm = preds\n        oof_gmm = overal_oof\n\n        RANDOM_SEED = np.random.randint(0, 1231231)\n\n        final_preds += preds / (ITERS)\n        final_preds_ranked += pd.DataFrame(preds).rank().values.reshape(final_preds_ranked.shape) / (ITERS)\n        final_auc += overal_oof / (ITERS)\n\nprint('\\n\\nfinal CV: ', roc_auc_score(overal_oof_y[min_idx:max_idx], final_auc[min_idx:max_idx]))\nsub = pd.read_csv('../input/instant-gratification/sample_submission.csv')\nsub['target'] = final_preds\nsub.to_csv('submission.csv',index=False)\n\nprint('\\n\\nfinal CV: ', roc_auc_score(overal_oof_y[min_idx:max_idx], final_auc[min_idx:max_idx]))\nsub = pd.read_csv('../input/instant-gratification/sample_submission.csv')\nsub['target'] = final_preds_ranked\nsub.to_csv('submission_ranked.csv',index=False)")


# In[ ]:




