#!/usr/bin/env python
# coding: utf-8

# # Summary
# The purpose of this kernel is that summarizing the result of several model output.  It seems that QDA and GMM had higher CV score than other models. Ensembling models got slightly better score but not boosting... so we need more explor something like make_classification method..
# 
# ## Single Models
# - [1. QDA with VarianceThreshold and StandardScaler](#1)
#    - CV: 0.96476
# - [2. QDA with PCA](#2)
#    - CV: 0.96457
# - [3. GaussianMixture with VarianceThreshold](#3)
#    - CV: 0.96748
# - [4. Logistic Regression with VarianceThreshold, PolynomialFeatures and StandardScaler](#4)
#    - CV: 0.95049
# - [5. LabelSpreading with VarianceThreshold](#5)
#    - CV: 0.93674
# - [6. kNN with VarianceThreshold and StandardScaler](#6)
#    - CV: 0.91593
# - [7. NN with PolynomialFeatures and StandardScaler](#7)
#    - CV: 0.94289
# 
# ## Single Model with Pseued Label
# - [P1. GSearch QDA with Pseued Label by GaussianMixture](#p1)
#    - CV: 0.96861
# 
# ## Blending
# - [AVG Blending](#b1)
#    - CV: 0.968017
# 
# ## Stacking
# - [Logistic Regression](#s1)
#     - CV: 0.969105

# ## Preparation

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.covariance import GraphicalLasso
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import warnings
import multiprocessing
from scipy.optimize import minimize  
warnings.filterwarnings('ignore')


# ## Load data

# In[ ]:


def load_data(data):
    return pd.read_csv(data)
    
with multiprocessing.Pool() as pool:
    train, test, sub = pool.map(load_data, ['../input/train.csv', '../input/test.csv', '../input/sample_submission.csv'])


# In[ ]:


cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# # Single Models

# <div id="1">
# </div>
# ## 1. QDA with VarianceThreshold and StandardScaler and get best paramater by GSearch
# - [VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html): Removes all low-variance features (under threshold)
# - [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): Standardize features by removing the mean and scaling to unit variance
# 
# 

# In[ ]:


oof_qda1 = np.zeros(len(train))
preds_qda1 = np.zeros(len(test))


# In[ ]:


params = np.arange(0.1, 0.5, 0.1) # [0.1 0.2, 0.3, 0.4, 0.5]
parameters = [{'reg_param': params}]
reg_params = np.zeros(512)


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in tqdm_notebook(range(512)):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==i]\n    test2 = test[test['wheezy-copper-turtle-magic']==i]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    test2.reset_index(drop=True, inplace=True)\n    \n    steps = [\n        ('vt', VarianceThreshold(threshold=2)),\n        ('sscaler', StandardScaler()),\n        #('rscaler', RobustScaler()),\n        #('mmscaler', MinMaxScaler())\n    ]\n    \n    pipe = Pipeline(steps=steps)\n    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])\n    data2 = pipe.fit_transform(data[cols])\n    train3 = data2[:train2.shape[0]]\n    test3 = data2[train2.shape[0]:]\n    \n    skf = StratifiedKFold(n_splits=11, random_state=42)\n    for train_index, test_index in skf.split(train2, train2['target']):\n\n        qda = QuadraticDiscriminantAnalysis()\n        clf = GridSearchCV(qda, parameters, cv=4)\n        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])\n        reg_params[i] = clf.best_params_['reg_param']\n        oof_qda1[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]\n        preds_qda1[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits")


# In[ ]:


print('QDA with VarianceThreshold and StandardScaler CV Score =',round(roc_auc_score(train['target'], oof_qda1), 5))


# In[ ]:


sub['target'] = preds_qda1
sub['target'].to_csv('submission_qda1.csv', index=False)


# In[ ]:


sub['target'] = preds_qda1
sub['target'].hist(bins=100, alpha=0.6)


# <div id="2">
# </div>
# ## 2. QDA with PCA

# In[ ]:


oof_qda2 = np.zeros(len(train))
preds_qda2 = np.zeros(len(test))


# In[ ]:


dict = dict()


# In[ ]:


get_ipython().run_cell_magic('time', '', "for s in range(512):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==s]\n    test2 = test[test['wheezy-copper-turtle-magic']==s]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    \n    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])\n    train3 = sel.transform(train2[cols])\n    test3 = sel.transform(test2[cols])\n        \n    dict[s] = train3.shape[1]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in tqdm_notebook(range(512)):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==i]\n    test2 = test[test['wheezy-copper-turtle-magic']==i]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    test2.reset_index(drop=True, inplace=True)  \n    \n    steps = [\n        #('pca', PCA(n_components=dict[i], random_state=42))\n        ('scaler', StandardScaler()), \n        ('fa', FeatureAgglomeration(n_clusters=dict[i]))\n    ]\n    \n    pipe = Pipeline(steps=steps)\n    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])\n    data2 = pipe.fit_transform(data[cols])\n    train3 = data2[:train2.shape[0]]\n    test3 = data2[train2.shape[0]:]\n\n    skf = StratifiedKFold(n_splits=11, random_state=42)\n    for train_index, test_index in skf.split(train2, train2['target']):\n\n        clf = QuadraticDiscriminantAnalysis(0.5)\n        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])\n        oof_qda2[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]\n        preds_qda2[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits")


# In[ ]:


print('QDA with PCA CV Score =',round(roc_auc_score(train['target'], oof_qda2), 5))


# In[ ]:


sub['target'] = preds_qda2
sub.to_csv('submission_qda2.csv', index=False)


# In[ ]:


sub['target'] = preds_qda2
sub['target'].hist(bins=100, alpha=0.6)


# <div id="3">
# </div>
# ## 3. GaussianMixture with VarianceThreshold

# In[ ]:


oof_gmm = np.zeros(len(train))
preds_gmm = np.zeros(len(test))


# In[ ]:


def get_mean_cov(x, y):
    
    model = GraphicalLasso()
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
    
    ms = np.stack([m1,m2])
    ps = np.stack([p1,p2])
    
    return ms, ps


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in tqdm_notebook(range(512)):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==i]\n    test2 = test[test['wheezy-copper-turtle-magic']==i]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    test2.reset_index(drop=True, inplace=True)\n    \n    steps = [\n        ('vt', VarianceThreshold(threshold=1.5))\n    ]\n    \n    pipe = Pipeline(steps=steps)\n    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])\n    data2 = pipe.fit_transform(data[cols])\n    train3 = data2[:train2.shape[0]]\n    test3 = data2[train2.shape[0]:]\n    \n    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)\n    for train_index, test_index in skf.split(train3, train2['target']):\n        \n        ms, ps = get_mean_cov(train3[train_index,:], train2.loc[train_index]['target'].values)\n        \n        clf = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001, reg_covar=0.001, max_iter=100, n_init=1, means_init=ms, precisions_init=ps)\n        clf.fit(np.concatenate([train3,test3],axis = 0))\n        oof_gmm[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,0]\n        preds_gmm[idx2] += clf.predict_proba(test3)[:,0] / skf.n_splits")


# In[ ]:


print('GaussianMixture with VarianceThreshold CV Score =',round(roc_auc_score(train['target'], oof_gmm), 5))


# In[ ]:


sub['target'] = preds_gmm
sub.to_csv('submission_gmm.csv', index=False)


# In[ ]:


sub['target'] = preds_gmm
sub['target'].hist(bins=100, alpha=0.6)


# <div id="4">
# </div>
# ## 4. Logistic Regression with VarianceThreshold, PolynomialFeatures and StandardScaler

# In[ ]:


oof_lr = np.zeros(len(train))
preds_lr = np.zeros(len(test))


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in tqdm_notebook(range(512)):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==i]\n    test2 = test[test['wheezy-copper-turtle-magic']==i]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    test2.reset_index(drop=True, inplace=True)\n    \n    steps = [\n        ('vt', VarianceThreshold(threshold=1.5)),\n        ('poly', PolynomialFeatures(degree=2)),\n        ('sc', StandardScaler())\n    ]\n    \n    pipe = Pipeline(steps=steps)\n    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])\n    data2 = pipe.fit_transform(data[cols])\n    train3 = data2[:train2.shape[0]]\n    test3 = data2[train2.shape[0]:]\n    \n    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)\n    for train_index, test_index in skf.split(train3, train2['target']):\n                \n        clf = LogisticRegression(solver='saga', penalty='l2', C=0.01, tol=0.001, n_jobs=-1)\n        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])\n        oof_lr[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]\n        preds_lr[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits")


# In[ ]:


print('Logistic Regression with VarianceThreshold CV Score =',round(roc_auc_score(train['target'], oof_lr), 5))


# In[ ]:


sub['target'] = preds_lr
sub.to_csv('submission_lr.csv', index=False)


# In[ ]:


sub['target'] = preds_lr
sub['target'].hist(bins=100, alpha=0.6)


# <div id="5">
# </div>
# ## 5. LabelSpreading with VarianceThreshold

# In[ ]:


oof_ls = np.zeros(len(train))
preds_ls = np.zeros(len(test))


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in tqdm_notebook(range(512)):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==i]\n    test2 = test[test['wheezy-copper-turtle-magic']==i]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    test2.reset_index(drop=True, inplace=True)\n    \n    steps = [\n        ('vt', VarianceThreshold(threshold=1.5))\n    ]\n    \n    pipe = Pipeline(steps=steps)\n    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])\n    data2 = pipe.fit_transform(data[cols])\n    train3 = data2[:train2.shape[0]]\n    test3 = data2[train2.shape[0]:]\n    \n    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)\n    for train_index, test_index in skf.split(train3, train2['target']):\n        \n        clf = LabelSpreading(gamma=0.01, kernel='rbf', max_iter=10)\n        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])\n        oof_ls[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]\n        preds_ls[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits")


# In[ ]:


print('LabelSpreading with VarianceThreshold CV Score =',round(roc_auc_score(train['target'], oof_ls), 5))


# In[ ]:


sub['target'] = preds_ls
sub.to_csv('submission_ls.csv', index=False)


# In[ ]:


sub['target'] = preds_ls
sub['target'].hist(bins=100, alpha=0.6)


# <div id="6">
# </div>
# ## 6. kNN with VarianceThreshold and StandardScaler

# In[ ]:


oof_knn = np.zeros(len(train))
preds_knn = np.zeros(len(test))


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in tqdm_notebook(range(512)):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==i]\n    test2 = test[test['wheezy-copper-turtle-magic']==i]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    test.reset_index(drop=True, inplace=True)\n    \n    steps = [\n        ('vt', VarianceThreshold(threshold=2)),\n        ('scaler', StandardScaler())\n    ]\n    \n    pipe = Pipeline(steps=steps)\n    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])\n    data2 = pipe.fit_transform(data[cols])\n    train3 = data2[:train2.shape[0]]\n    test3 = data2[train2.shape[0]:]\n    \n    skf = StratifiedKFold(n_splits=11, random_state=42)\n    for train_index, test_index in skf.split(train2, train2['target']):\n\n        clf = KNeighborsClassifier(n_neighbors=17, p=2.9, n_jobs=-1)\n        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])\n        oof_knn[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]\n        preds_knn[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits")


# In[ ]:


print('kNN with VarianceThreshold and StandardScaler CV Score =',round(roc_auc_score(train['target'], oof_knn), 5))


# In[ ]:


sub['target'] = preds_knn
sub.to_csv('submission_knn.csv', index=False)


# In[ ]:


sub['target'] = preds_knn
sub['target'].hist(bins=100, alpha=0.6)


# <div id="7">
# </div>
# ## 7. NN with PolynomialFeatures and StandardScaler

# In[ ]:


oof_nn = np.zeros(len(train))
preds_nn = np.zeros(len(test))


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in tqdm_notebook(range(512)):\n\n    train2 = train[train['wheezy-copper-turtle-magic']==i]\n    test2 = test[test['wheezy-copper-turtle-magic']==i]\n    idx1 = train2.index\n    idx2 = test2.index\n    train2.reset_index(drop=True, inplace=True)\n    test2.reset_index(drop=True, inplace=True)\n    \n    steps = [\n        ('vt', VarianceThreshold(threshold=1.5)),\n        ('poly', PolynomialFeatures(degree=2)),\n        ('sc', StandardScaler())\n        # log scale\n    ]\n    \n    pipe = Pipeline(steps=steps)\n    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])\n    data2 = pipe.fit_transform(data[cols])\n    train3 = data2[:train2.shape[0]]\n    test3 = data2[train2.shape[0]:]\n    \n    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)\n    for train_index, test_index in skf.split(train3, train2['target']):\n                \n        clf = MLPClassifier(random_state=3, activation='relu', solver='lbfgs', tol=1e-06, hidden_layer_sizes=(250, ))\n        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])\n        oof_nn[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]\n        preds_nn[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits")


# In[ ]:


print('NN with PolynomialFeatures and StandardScaler CV Score =',round(roc_auc_score(train['target'], oof_nn), 5))


# In[ ]:


sub['target'] = preds_nn
sub.to_csv('submission_nn.csv', index=False)


# In[ ]:


sub['target'] = preds_nn
sub['target'].hist(bins=100, alpha=0.6)


# <div id="p1">
# </div>
# ## P1. GSearch QDA with Pseued Label by GaussianMixture

# In[ ]:


oof_qda3 = np.zeros(len(train))
preds_qda3 = preds_gmm.copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in range(10):\n    \n    test[\'target\'] = preds_qda3\n    test.loc[test[\'target\'] > 0.955, \'target\'] = 1\n    test.loc[test[\'target\'] < 0.045, \'target\'] = 0\n    usefull_test = test[(test[\'target\'] == 1) | (test[\'target\'] == 0)]\n    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)\n    \n    print(usefull_test.shape[0], "Test Record added for iteration : ", i + 1)\n    \n    new_train.loc[oof_qda3 > 0.995, \'target\'] = 1\n    new_train.loc[oof_qda3 < 0.005, \'target\'] = 0\n    \n    oof_qda3 = np.zeros(len(train))\n    preds_qda = np.zeros(len(test))\n    \n    for i in tqdm_notebook(range(512)):\n\n        train2 = new_train[new_train[\'wheezy-copper-turtle-magic\']==i]\n        test2 = test[test[\'wheezy-copper-turtle-magic\']==i]\n        idx1 = train[train[\'wheezy-copper-turtle-magic\']==i].index\n        idx2 = test2.index\n        train2.reset_index(drop=True, inplace=True)\n        test2.reset_index(drop=True, inplace=True)\n        \n        steps = [\n            (\'vt\', VarianceThreshold(threshold=1.5)),\n            (\'scaler\', StandardScaler())\n        ]\n        \n        pipe = Pipeline(steps=steps)\n        train3 = pipe.fit_transform(train2[cols])\n        test3 = pipe.fit_transform(test2[cols])\n        \n        skf = StratifiedKFold(n_splits=11, random_state=42)\n        for train_index, test_index in skf.split(train2, train2[\'target\']):\n            oof_test_index = [t for t in test_index if t < len(idx1)]\n            \n            clf = QuadraticDiscriminantAnalysis(reg_params[i])\n            clf.fit(train3[train_index,:],train2.loc[train_index][\'target\'])\n            if len(oof_test_index) > 0:\n                oof_qda3[idx1[oof_test_index]] = clf.predict_proba(train3[oof_test_index,:])[:,1]\n            preds_qda3[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits\n    \n    print(\'QDA with Pseued Label by GaussianMixture CV Score =\',round(roc_auc_score(train[\'target\'], oof_qda3), 5))')


# In[ ]:


sub['target'] = preds_qda3
sub.to_csv('submission_gda3.csv', index=False)


# In[ ]:


sub['target'] = preds_qda3
sub['target'].hist(bins=100, alpha=0.6)


# <div id="p2">
# </div>
# ## P2. GSearch QDA with Pseued Label by QDA

# In[ ]:


oof_qda4 = oof_qda1.copy()
preds_qda4 = preds_qda1.copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i in range(10):\n    \n    test[\'target\'] = preds_qda4\n    test.loc[test[\'target\'] > 0.955, \'target\'] = 1\n    test.loc[test[\'target\'] < 0.045, \'target\'] = 0\n    usefull_test = test[(test[\'target\'] == 1) | (test[\'target\'] == 0)]\n    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)\n    \n    print(usefull_test.shape[0], "Test Record added for iteration : ", i + 1)\n    \n    new_train.loc[oof_qda4 > 0.995, \'target\'] = 1\n    new_train.loc[oof_qda4 < 0.005, \'target\'] = 0\n    \n    oof_qda4= np.zeros(len(train))\n    preds_qd4 = np.zeros(len(test))\n    \n    for i in tqdm_notebook(range(512)):\n\n        train2 = new_train[new_train[\'wheezy-copper-turtle-magic\']==i]\n        test2 = test[test[\'wheezy-copper-turtle-magic\']==i]\n        idx1 = train[train[\'wheezy-copper-turtle-magic\']==i].index\n        idx2 = test2.index\n        train2.reset_index(drop=True, inplace=True)\n        test2.reset_index(drop=True, inplace=True)\n        \n        steps = [\n            (\'vt\', VarianceThreshold(threshold=1.5)),\n            (\'scaler\', StandardScaler())\n        ]\n        \n        pipe = Pipeline(steps=steps)\n        train3 = pipe.fit_transform(train2[cols])\n        test3 = pipe.fit_transform(test2[cols])\n        \n        skf = StratifiedKFold(n_splits=11, random_state=42)\n        for train_index, test_index in skf.split(train2, train2[\'target\']):\n            oof_test_index = [t for t in test_index if t < len(idx1)]\n            \n            clf = QuadraticDiscriminantAnalysis(reg_params[i])\n            clf.fit(train3[train_index,:],train2.loc[train_index][\'target\'])\n            if len(oof_test_index) > 0:\n                oof_qda4[idx1[oof_test_index]] = clf.predict_proba(train3[oof_test_index,:])[:,1]\n            preds_qda4[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits\n    \n    print(\'GSearch QDA with Pseued Label by QDA CV Score =\',round(roc_auc_score(train[\'target\'], oof_qda3), 5))')


# In[ ]:


sub['target'] = preds_qda4
sub.to_csv('submission_gda3.csv', index=False)


# In[ ]:


sub['target'] = preds_qda4
sub['target'].hist(bins=100, alpha=0.6)


# # Ensembling

# In[ ]:


print('All Model CV Score Summary')
print('---------------------------------')
print('1. QDA with VarianceThreshold and StandardScaler', roc_auc_score(train['target'], oof_qda1))
print('2. QDA with PCA', roc_auc_score(train['target'], oof_qda2))
print('3. GaussianMixture with VarianceThreshold', roc_auc_score(train['target'], oof_gmm))
print('4. Logistic Regression with PolynomialFeatures and StandardScaler', roc_auc_score(train['target'], oof_lr))
print('5. LabelSpreading with VarianceThreshold', roc_auc_score(train['target'], oof_ls))
print('6. kNN with VarianceThreshold and StandardScaler', roc_auc_score(train['target'], oof_knn))
print('P1. QDA with Pseued Label by GaussianMixture', roc_auc_score(train['target'], oof_qda3))


# In[ ]:


preds_list = [preds_qda1, preds_qda2, preds_gmm, preds_lr, preds_ls, preds_knn, preds_nn, preds_qda3]
oof_list = [oof_qda1, oof_qda2, oof_gmm, oof_lr, oof_ls, oof_knn, oof_nn, oof_qda3]
print('The number of model is {}'.format(len(preds_list)))


# <div id="b1">
# </div>
# ## AVG Blending

# In[ ]:


oof_avg = sum(oof_list) / len(oof_list)
preds_avg = sum(preds_list) / len(preds_list)


# In[ ]:


print('{} model blend CV score ='.format(len(preds_list)),round(roc_auc_score(train['target'], oof_avg),6))


# In[ ]:


sub['target'] = preds_avg
sub.to_csv('submission_blend_avg.csv', index=False)


# In[ ]:


sub['target'] = preds_avg
sub['target'].hist(bins=100, alpha=0.6)


# ## Stacking Model Blending

# In[ ]:


oof_blend = np.zeros(len(train))
preds_knn = np.zeros(len(test))


# In[ ]:


oof_blend = 0.5*(oof_qda3 + oof_qda4)
preds_blend = 0.5*(preds_qda3 + preds_qda4)


# In[ ]:


print('Stacking model blend CV score =',round(roc_auc_score(train['target'], oof_blend),6))


# In[ ]:


sub['target'] = preds_blend
sub.to_csv('submission_blend.csv', index=False)


# In[ ]:


sub['target'] = preds_blend
sub['target'].hist(bins=100, alpha=0.6)


# ## Stacking

# <div id="s1">
# </div>    
# ## Logistic Regression Stacking

# In[ ]:


oof_qda1 = oof_qda1.reshape(-1, 1)
oof_qda2 = oof_qda2.reshape(-1, 1)
oof_gmm = oof_gmm.reshape(-1, 1)
oof_lr = oof_lr.reshape(-1, 1)
oof_ls = oof_ls.reshape(-1, 1)
oof_knn = oof_knn.reshape(-1, 1)
oof_nn = oof_nn.reshape(-1, 1)
oof_qda3 = oof_qda3.reshape(-1, 1)


# In[ ]:


preds_qda1 = preds_qda1.reshape(-1, 1)
preds_qda2 = preds_qda2.reshape(-1, 1)
preds_gmm = preds_gmm.reshape(-1, 1)
preds_lr = preds_lr.reshape(-1, 1)
preds_ls = preds_ls.reshape(-1, 1)
preds_knn = preds_knn.reshape(-1, 1)
preds_nn = preds_nn.reshape(-1, 1)
preds_qda3 = preds_qda3.reshape(-1, 1)


# In[ ]:


train_stck = np.concatenate([oof_qda1, oof_qda2, oof_gmm, oof_lr, oof_ls, oof_knn, oof_nn, oof_qda3], axis=1)
test_stack = np.concatenate([preds_qda1, preds_qda2, preds_gmm, preds_lr, preds_ls, preds_knn, preds_nn, preds_qda3], axis=1)


# In[ ]:


oof_stack = np.zeros(len(train)) 
pred_stack = np.zeros(len(test))


# In[ ]:


for train_index, test_index in skf.split(train_stck, train['target']):
    clf = LogisticRegression(solver='saga', penalty='l2', C=0.01, tol=0.001)
    clf.fit(train_stck[train_index], train['target'][train_index])
    oof_stack[test_index] = clf.predict_proba(train_stck[test_index,:])[:,1]
    pred_stack += clf.predict_proba(test_stack)[:,1] / skf.n_splits


# In[ ]:


print('{} model stack CV score ='.format(len(preds_list)),round(roc_auc_score(train['target'], oof_stack),6))


# In[ ]:


sub['target'] = pred_stack
sub.to_csv('submission_stack.csv', index=False)


# In[ ]:


sub['target'] = pred_stack
sub['target'].hist(bins=100, alpha=0.6)

