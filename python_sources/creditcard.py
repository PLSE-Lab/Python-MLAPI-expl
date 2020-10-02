import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

def getCvMetrics(cfr, X, y):    
    #Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    print('Iterating through the training and cross validation sets...')
    ploss = []
    aucs = []
    for train, cval in cv.split(X, y):
        cfr.fit(X[train], y[train])
        loss = log_loss(y[cval], cfr.predict_proba(X[cval]))
        auc = roc_auc_score(y[cval], cfr.predict_proba(X[cval])[:, 1])
        print(loss, auc)
        ploss.append(loss)
        aucs.append(auc)

    #print out the mean of the cross-validated results
    print('Mean log-loss: %f. Mean AUC %f' % 
                            (np.array(ploss).mean(), np.array(aucs).mean()))

print('Loading data...')
dfcard = pd.read_csv('../input/creditcard.csv')
print(dfcard.shape)
# print(dfcard.head())
# print(dfcard.describe())
# print(dfcard.info())

# print('Fraud times')
# print(dfcard.loc[dfcard.Class==1, 'Time'].describe())
# print('No Fraud times')
# print(dfcard.loc[dfcard.Class==0, 'Time'].describe())

fraud = dfcard.Class
print('Fraud ratio: %f' % fraud.mean())
print(fraud.value_counts())

train=dfcard.drop(['Class'],axis=1)
print('Train set shape {}'.format(train.shape))
print(train.columns)

print('Training...')
Xtrain, Xval, ytrain, yval = train_test_split(train, fraud, test_size=0.2, stratify=fraud, 
                                                                    random_state=0)
clf = xgboost.XGBClassifier(max_depth=6, learning_rate = 0.05, 
                #subsample = 0.9, colsample_bytree = 0.9, 
                n_estimators=100, base_score=0.0017, nthread=-1) #0.97172
clf = GradientBoostingClassifier(max_depth=6, learning_rate = 0.05, 
                #subsample = 0.9, colsample_bytree = 0.9, 
                n_estimators=100) # 0.867587
clf = ExtraTreesClassifier(n_estimators=200, class_weight='balanced', max_depth=7, 
                            random_state=12, n_jobs=-1) # 0.982360
clf = LogisticRegression(C=1000, class_weight='balanced')
print('Val AUC: %f'%roc_auc_score(yval, clf.fit(Xtrain, ytrain).predict_proba(Xval)[:,1]))
# getCvMetrics(clf, train.values, fraud) 
# xgb: 0.982599 [0.9871420, 0.988217, 0.983688, 0.983114, 0.970831]
# etrees: 0.979302 [0.990166, 0.982376, 0.978804, 0.976500, 0.968663]
scores = cross_val_score(clf, train.values, fraud, cv=5, scoring='roc_auc')
print('Validation AUCs (5-fold)')
print(scores.mean(), scores) 
# xgb: 0.971972 [ 0.98227762  0.95253806  0.95207493  0.98965631  0.98331328]