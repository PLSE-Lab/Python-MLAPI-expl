#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# LOAD LIBRARIES
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.gaussian_process.kernels as ker
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd, os, gc
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLasso
from sklearn.mixture import GaussianMixture


# In[ ]:


def get_mean_cov(x,y):
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
    return ms,ps


# In[ ]:


print('Reading Train Data...')
train = pd.read_csv('../input/train.csv')
print('Reading Test Data...')
test = pd.read_csv('../input/test.csv')
print('Finish Reading.')


# In[ ]:


n_folds = 11


# In[ ]:


# INITIALIZE VARIABLES
oof_GMM = np.zeros(len(train))
preds_GMM = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in tqdm_notebook(range(512)):
    
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=n_folds, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL WITH GMM
        ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values) 
        gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)
        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))
        oof_GMM[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:,0]
        preds_GMM[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits
        
    if i%64==0:     
        print(i, 'GMM oof auc : ', round(roc_auc_score(train['target'][idx1], oof_GMM[idx1]), 5))


# In[ ]:


auc_GMM = roc_auc_score(train['target'],oof_GMM)
print('GMM scores CV: ',round(auc_GMM, 5))


# In[ ]:


# INITIALIZE VARIABLES
test['target'] = preds_GMM
oof_QDA = np.zeros(len(train))
preds_QDA = np.zeros(len(test))

oof_NuSVC = np.zeros(len(train))
preds_NuSVC = np.zeros(len(test))

oof_KNN = np.zeros(len(train))
preds_KNN = np.zeros(len(test))

oof_MLP = np.zeros(len(train))
preds_MLP = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for k in tqdm_notebook(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==k] 
    train2p = train2.copy(); idx1 = train2.index 
    test2 = test[test['wheezy-copper-turtle-magic']==k]
    idx2 = test2.index 
    
    # ADD PSEUDO LABEL DATA
    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()
    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1
    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 
    train2p = pd.concat([train2p,test2p],axis=0)
    train2p.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     
    train3p = sel.transform(train2p[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    pca = PCA(svd_solver='full',n_components='mle').fit(train2p[cols])
    train4p = pca.transform(train2p[cols])
    train4 = pca.transform(train2[cols])
    test4 = pca.transform(test2[cols])
    sc1 = StandardScaler()
    train4p = sc1.fit_transform(train4p)
    train4 = sc1.transform(train4)
    test4 = sc1.transform(test4)
    
    poly = PolynomialFeatures().fit(train3p)
    train5p = poly.transform(train3p)
    train5 = poly.transform(train3)
    test5 = poly.transform(test3)
    sc2 = StandardScaler()
    train5p = sc2.fit_transform(train5p)
    train5 = sc2.transform(train5)
    test5 = sc2.transform(test5)
        
    # STRATIFIED K FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3p, train2p['target']):
        test_index3 = test_index[ test_index<len(train3) ] # ignore psuedo in oof
        
        # MODEL AND PREDICT WITH QDA
        clf_QDA = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf_QDA.fit(train3p[train_index,:],train2p.loc[train_index]['target'])
        oof_QDA[idx1[test_index3]] = clf_QDA.predict_proba(train3[test_index3,:])[:,1]
        preds_QDA[idx2] += clf_QDA.predict_proba(test3)[:,1] / skf.n_splits
        
        clf_NuSVC = NuSVC(probability=True,kernel='poly',degree=4,gamma='auto',nu=0.59, coef0=0.053)
        clf_NuSVC.fit(train4p[train_index,:],train2p.loc[train_index]['target'])
        oof_NuSVC[idx1[test_index3]] = clf_NuSVC.predict_proba(train4[test_index3,:])[:,1]
        preds_NuSVC[idx2] += clf_NuSVC.predict_proba(test4)[:,1] / skf.n_splits
        
        clf_KNN = KNeighborsClassifier(n_neighbors=10,weights='distance',p=2)
        clf_KNN.fit(train3p[train_index,:],train2p.loc[train_index]['target'])
        oof_KNN[idx1[test_index3]] = clf_KNN.predict_proba(train3[test_index3,:])[:,1]
        preds_KNN[idx2] += clf_KNN.predict_proba(test3)[:,1] / skf.n_splits
        
        clf_MLP = MLPClassifier(activation='relu', solver='lbfgs', tol=1e-06, hidden_layer_sizes=(250,), random_state=42)
        clf_MLP.fit(train5p[train_index,:],train2p.loc[train_index]['target'])
        oof_MLP[idx1[test_index3]] = clf_MLP.predict_proba(train5[test_index3,:])[:,1]
        preds_MLP[idx2] += clf_MLP.predict_proba(test5)[:,1] / skf.n_splits
        
    if k%64==0:     
        print(k, 'QDA oof auc : ', round(roc_auc_score(train['target'][idx1], oof_QDA[idx1]), 5))
        print(k, 'NuSVC oof auc : ', round(roc_auc_score(train['target'][idx1], oof_NuSVC[idx1]), 5))
        print(k, 'KNN oof auc : ', round(roc_auc_score(train['target'][idx1], oof_KNN[idx1]), 5))
        print(k, 'MLP oof auc : ', round(roc_auc_score(train['target'][idx1], oof_MLP[idx1]), 5))


# In[ ]:


# PRINT CV AUC
auc_QDA = roc_auc_score(train['target'],oof_QDA)
print('Pseudo Labeled QDA scores CV: ',round(auc_QDA,5))

auc_NuSVC = roc_auc_score(train['target'],oof_NuSVC)
print('Pseudo Labeled NuSVC scores CV: ',round(auc_NuSVC,5))

auc_KNN = roc_auc_score(train['target'],oof_KNN)
print('Pseudo Labeled KNN scores CV: ',round(auc_KNN,5))

auc_MLP = roc_auc_score(train['target'],oof_MLP)
print('Pseudo Labeled MLP scores CV: ',round(auc_MLP,5))


# In[ ]:


train_new = pd.DataFrame(np.concatenate((oof_NuSVC.reshape(-1,1),oof_KNN.reshape(-1,1),oof_QDA.reshape(-1,1),oof_MLP.reshape(-1,1)), axis=1))
test_new = pd.DataFrame(np.concatenate((preds_NuSVC.reshape(-1,1),preds_KNN.reshape(-1,1),preds_QDA.reshape(-1,1),preds_MLP.reshape(-1,1)), axis=1))


# In[ ]:


param = {
    'bagging_freq': 3,
    'bagging_fraction': 0.8,
    'boost_from_average':'False',
    'boost': 'gbdt',
    'feature_fraction': 1,
    'learning_rate': 0.05,
    'max_depth': 10,
    'metric':'auc',
    'min_data_in_leaf': 82,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'objective': 'binary', 
    'verbosity': 1,
    'seed': 42
}

import lightgbm as lgb
N = 5
skf_lgb = StratifiedKFold(n_splits=N, random_state=42)

oof_lgb = np.zeros(train_new.shape[0])
pred_stack = np.zeros(len(test_new))

for fold_, (trn_idx, val_idx) in enumerate(skf_lgb.split(train_new, train['target'])):
    print("Fold {}".format(fold_+1))
    x_train, y_train = train_new.iloc[trn_idx], train['target'].iloc[trn_idx]
    x_val, y_val = train_new.iloc[val_idx], train['target'].iloc[val_idx]
    x_train.head()
    
    trn_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    classifier = lgb.train(param, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 200)

    val_pred = classifier.predict(x_val, num_iteration=classifier.best_iteration)
    oof_lgb[val_idx] = val_pred
    pred_stack += classifier.predict(test_new, num_iteration=classifier.best_iteration) / N
    print(roc_auc_score(y_val, val_pred))

auc_lgb = roc_auc_score(train['target'],oof_lgb)
print('LGB auc: ',round(auc_lgb,5))


# In[ ]:


submission_stack = pd.read_csv("../input/sample_submission.csv")
submission_stack['target'] = pred_stack
submission_stack.head()
submission_stack.to_csv('submission.csv', index=False)


# In[ ]:


train_new.to_csv('train_new.csv', index=False)
test_new.to_csv('test_new.csv', index=False)

