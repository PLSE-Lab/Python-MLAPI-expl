#!/usr/bin/env python
# coding: utf-8

# ### Loading Libraries

# In[ ]:


from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold


# ### Loading Data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')")


# ## Identifying Clusters and Predicting Classes
# 
# In what follows, I will assume that there are 2 clusters per class in the data set. To identify these clusters let's run `GMM` on positive and negative instances separately. In each case, our goal is to label instances that belong to two different clusters. We will strore the mean and covariance matricies for all four clusters for future use.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nNFOLDS=11\nRS=42\n\noof=np.zeros(len(train))\npreds=np.zeros(len(test))\n\nmagic_max=train[\'wheezy-copper-turtle-magic\'].max()\nmagic_min=train[\'wheezy-copper-turtle-magic\'].min()\n\nauc_all=np.array([])\n\nprint(f"Computing centroids and covariances for the four clusters (two per class).")\n\n# BUILD 512 SEPARATE NON-LINEAR MODELS\n#for i in tqdm(range(10)): \nfor i in tqdm(range(magic_min, magic_max+1)):  \n    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS i     \n    X = train[train[\'wheezy-copper-turtle-magic\']==i].copy()\n    Y = X.pop(\'target\').values\n    X_test = test[test[\'wheezy-copper-turtle-magic\']==i].copy()\n    idx_train = X.index \n    idx_test = X_test.index\n    X.reset_index(drop=True,inplace=True)\n\n    cols = [c for c in X.columns if c not in [\'id\', \'wheezy-copper-turtle-magic\']]\n\n    X = X[cols].values             # numpy.ndarray\n    X_test = X_test[cols].values   # numpy.ndarray\n\n    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)\n    vt = VarianceThreshold(threshold=1.5).fit(X)\n    X = vt.transform(X)            # numpy.ndarray\n    X_test = vt.transform(X_test)  # numpy.ndarray   \n\n    # STRATIFIED K FOLD\n    auc_folds=np.array([])\n    \n    folds = StratifiedKFold(n_splits=NFOLDS, random_state=RS)\n\n    for fold_num, (train_index, val_index) in enumerate(folds.split(X, Y), 1):\n\n        X_train, Y_train = X[train_index, :], Y[train_index]\n        X_val, Y_val = X[val_index, :], Y[val_index]\n\n        X_train_0 = X_train[Y_train==0]\n        Y_train_0 = Y_train[Y_train==0].reshape(-1, 1)\n\n        X_train_1 = X_train[Y_train==1]\n        Y_train_1 = Y_train[Y_train==1].reshape(-1, 1)\n\n        params={\'n_components\' : 2, \n                \'init_params\': \'random\', \n                \'covariance_type\': \'full\', \n                \'tol\':0.001, \n                \'reg_covar\': 0.001,#0.001, \n                \'max_iter\': 100, \n                \'n_init\': 10,\n               }\n\n        clf_0 = GaussianMixture(**params)\n\n        clf_0.fit(X_train_0)\n        #labels_0 = clf_0.predict(X_train_0)\n        means_0 = clf_0.means_\n        covs_0 = clf_0.covariances_\n        ps_0 = [np.linalg.inv(m) for m in covs_0]\n        \n        clf_1 = GaussianMixture(**params)\n        \n        clf_1.fit(X_train_1)\n        #labels_1 = clf_1.predict(X_train_1)\n        means_1 = clf_1.means_\n        covs_1 = clf_1.covariances_\n        ps_1 = [np.linalg.inv(m) for m in covs_1]\n        \n        #MEANS AND COVARIANCES FOR THE CLUSTERS       \n        ms = np.stack((means_0[0], means_0[1], means_1[0], means_1[1]))\n        ps = np.stack((ps_0[0], ps_0[1], ps_1[0], ps_1[1]))\n        \n        #PARAMETERS FOR THE MAIN CLASSIFIER\n        params={\'n_components\' : 4, \n                \'init_params\': \'random\', \n                \'covariance_type\': \'full\', \n                \'tol\':0.001, \n                \'reg_covar\': 0.001, \n                \'max_iter\': 100, \n                \'n_init\': 10, \n                \'means_init\': ms, \n                \'precisions_init\': ps,\n               }\n        \n        #INSTANTIATING THE MAIN CLASSIFIER\n        clf = GaussianMixture(**params)   \n        \n        clf.fit(np.concatenate([X_train, X_test], axis = 0))\n        \n        oof[idx_train[val_index]] = np.sum(clf.predict_proba(X_val)[:, 2:], axis=1)\n        preds[idx_test] += np.sum(clf.predict_proba(X_test)[:,2: ], axis=1)/NFOLDS\n        \n#         oof[idx_train[val_index]] = clf.predict_proba(X_val)[:,1]\n#         preds[idx_test] += clf.predict_proba(X_test)[:,1]/NFOLDS\n\n        auc = roc_auc_score(Y_val, oof[idx_train[val_index]])\n        auc_folds = np.append(auc_folds, auc)\n\n    auc_all = np.append(auc_all, np.mean(auc_folds))\n\n# PRINT CROSS-VALIDATION AUC FOR THE CLASSFIER\nauc_combo = roc_auc_score(train[\'target\'].values, oof)\nauc_folds_average = np.mean(auc_all)\nstd = np.std(auc_all)/(np.sqrt(NFOLDS)*np.sqrt(magic_max+1))\n\nprint(f\'The combined AUC CV score is {round(auc_combo,5)}.\')    \nprint(f\'The folds average AUC CV score is {round(auc_folds_average,5)}.\')\nprint(f\'The standard deviation is {round(std, 5)}.\')')


# ### Creating the submission file

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)
