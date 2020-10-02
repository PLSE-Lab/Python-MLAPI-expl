

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from AlternateLinearModel import AlternateLasso
from sklearn.model_selection import (train_test_split,GridSearchCV, cross_val_score,StratifiedKFold)
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, auc,matthews_corrcoef,classification_report, roc_curve, cohen_kappa_score, make_scorer,accuracy_score,roc_auc_score,precision_score,recall_score
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,FunctionTransformer
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from numpy import tanh
from sklearn.pipeline import Pipeline

data=np.array(pd.read_csv("../input/Financial Distress.csv").values)
X=data[:,np.r_[:,3:128]]
y=data[:,np.r_[:,2]]
y=np.array([0 if i > -0.50 else 1 for i in y])


weights = [7,
 8,
 9,
 13,
 18,
 25,
 40,
 48,
 50,
 54,
 57,
 59,
 71,
 75,
 84,
 93,
 94,
 96,
 97,
 101,
 106,
 113]
X = X[:,weights]

from sklearn.preprocessing import scale,PolynomialFeatures
#X = scale(X)
poly = PolynomialFeatures(degree = 2)
X = poly.fit_transform(X)




# # Load Data
# data=np.array(pd.read_csv('Financial Distress.csv').values)
# X=data[:,np.r_[:,3:128]]
# y=data[:,np.r_[:,2]]
# y=np.array([0 if i > -0.50 else 1 for i in y])


# # Divide Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# # Compute Cohen's Kappa or Auc as scoring criterion due to imbalanced data set
kappa_scorer = make_scorer(cohen_kappa_score)
auc_scorer=make_scorer(roc_auc_score)
F_measure_scorer = make_scorer(f1_score)

# # Use tanh to Preprocess Features
ft=FunctionTransformer(tanh)

# # Use XGBClassifier as base_estimator
xg=XGBClassifier(learning_rate =0.03,n_jobs=16,
                 n_estimators=10,
                 max_depth=3,
                 min_child_weight=1,
                 # gamma=0,
                 # subsample=0.8,
                 # colsample_bytree=0.8,
                 objective= 'binary:logistic',
                 scale_pos_weight=0.17,
                 # seed=27,
                 # base_score=0.51,
                 # reg_lambda=0.1,
                 # colsample_bylevel=1
                 )

# # Create Pipeline
pipeline = Pipeline(steps=[('scaler', ft),
                           ('clf', BalancedBaggingClassifier(
                               base_estimator=xg,
                               ratio='auto',
                              #replacement=False,
                               random_state=42,
                               max_features=5,
                               #max_samples=20
                                ))])
# # Determine Hyper-parameters to be Tuned
param_grid = {#'clf__ base_estimator__scale_pos_weight':[0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30],
              'clf__n_estimators':range(8,40,4),
              'clf__max_features' :[0.6,0.7,0.8, 0.9,1],
               }

cv=StratifiedKFold(n_splits=5,random_state=42)

# # Grid Search to Tune Hyper-parameters
xg_cv = GridSearchCV(pipeline, param_grid, cv=cv, scoring='recall')
xg_cv.fit(X_train, y_train)
print("Tuned xg_cv best params: {}".format(xg_cv.best_params_))

print("Train Results:")
ypred_train = xg_cv.predict(X_train)
print('MCC',matthews_corrcoef(y_train,ypred_train))
print(confusion_matrix(y_train, ypred_train))
print(classification_report(y_train, ypred_train))

print('######################')

# # Testing the fitted model on test data

print("Test Results:")

ypred_test = xg_cv.predict(X_test)
yprob_test= xg_cv.predict_proba(X_test)

print('MCC',matthews_corrcoef(y_test,ypred_test))
print(confusion_matrix(y_test, ypred_test))
print(classification_report(y_test, ypred_test))

# # Calculate some performance measures using test data: Accuracy, Auc, F-measure, Cohen's kappa, Precision, Recall

print('Accuracy:', accuracy_score(y_test, ypred_test))
print('AUC:', roc_auc_score(y_test,yprob_test[:,1]))
print('F1 Measure:',f1_score(y_test,ypred_test))
print('Cohen Kappa:',cohen_kappa_score(y_test, ypred_test))
print('Precision:',precision_score(y_test,ypred_test))
print('recall:',recall_score(y_test,ypred_test))


