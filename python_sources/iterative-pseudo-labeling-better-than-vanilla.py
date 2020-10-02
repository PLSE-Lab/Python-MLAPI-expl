import time
import random
import numpy as np, pandas as pd, os
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer, PowerTransformer, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer, PowerTransformer, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.svm import NuSVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import multiprocessing
# loads of unused stuff imported ;) 
import warnings
warnings.filterwarnings('ignore')

with multiprocessing.Pool() as pool: 
    train, test, sub = pool.map(load_data, ['../input/train.csv', '../input/test.csv','../input/sample_submission.csv'])

models = [QuadraticDiscriminantAnalysis(0.2,reg_param=0.111,tol=1.0e-4)]

model=models[0]

# make list of the 'magic' feature values
mvals=train['wheezy-copper-turtle-magic'].unique().tolist()

cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))
preds = np.zeros(len(test))

for i in mvals:
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    sel = VarianceThreshold(threshold=2).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        clf = model
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits


# PRINT CV AUC
print('FIRST PASS: FIT TRAIN AND PREDICT TEST')
auc = roc_auc_score(train['target'],oof)
print('scores CV =',round(auc,5),'(train only)')

# pseudo labeling (first pass)
test['target']=preds
test.loc[test['target']>=0.5,'target'] = 1
test.loc[test['target']<0.5,'target'] = 0

preds_test=preds
oof_old = oof

# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')

#REPEAT THE ABOVE BUT MODEL TRAIN + pseudo labeled TEST, THEN predict TEST

traina=pd.concat([train,test],axis=0)
traina.reset_index(drop=True,inplace=True)

oof = np.zeros(len(traina))
preds = np.zeros(len(test))

for i in mvals:
    
    train2 = traina[traina['wheezy-copper-turtle-magic']==i].copy()
    test2 = test[test['wheezy-copper-turtle-magic']==i].copy()
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
 
    sel = VarianceThreshold(threshold=2).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
   
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):

        clf = model
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        
# PRINT CV AUC
print('FIT TRAIN + PSEUDO LABELED TEST')
auc = roc_auc_score(traina['target'],oof)
print('scores CV =',round(auc,5),'(train + pseudo labeling)')

# iterate the pseudo labeling to increase model accuracy
for _ in range(5):
    
    #REPEAT THE ABOVE USING NEW TEST PREDS TO RELABEL TEST 

    test['target']=preds
    test.loc[test['target']>=0.5,'target'] = 1
    test.loc[test['target']<0.5,'target'] = 0
    traina=pd.concat([train,test],axis=0)

    traina.reset_index(drop=True,inplace=True)
    
    oof = np.zeros(len(traina))
    preds = np.zeros(len(test))

    for i in mvals:
        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS i
        train2 = traina[traina['wheezy-copper-turtle-magic']==i].copy()
        test2 = test[test['wheezy-copper-turtle-magic']==i].copy()
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
        sel = VarianceThreshold(threshold=2).fit(train2[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        # STRATIFIED K-FOLD
        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3, train2['target']):

            # MODEL AND PREDICT WITH QDA
            clf = model
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

    # PRINT MESSAGE
    print(f'FIT TRAIN + PSEUDO LABELED TEST (ITERATION)')
    auc = roc_auc_score(traina['target'],oof)
    print('scores CV =',round(auc,5),'(train + pseudo labeling)')


sub['target'] = preds
sub.to_csv('submission.csv',index=False)