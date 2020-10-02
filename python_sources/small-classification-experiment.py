# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

def main():
    df = pd.read_csv('../input/all_data_R.csv')
    cols_to_transform = ["sex", "corpus"]
    df2 = pd.get_dummies(df, columns=cols_to_transform)
    all_features = list(df2.columns)
    features_to_leave_out = ["Y", "filename", "age", "group"]
    features = [i for i in all_features if i not in features_to_leave_out]

    X = df2[features]
    y = df2['Y']

    rf = RandomForestClassifier(max_depth=5, n_estimators=10, class_weight="balanced")
    abf = AdaBoostClassifier()
    xgb = XGBClassifier()

    cv = StratifiedKFold(n_splits=50)

    mean_tpr_rf = 0.0
    mean_fpr_rf = np.linspace(0, 1, 100)
    mean_tpr_abf = 0.0
    mean_fpr_abf = np.linspace(0, 1, 100)
    mean_tpr_xgb = 0.0
    mean_fpr_xgb = np.linspace(0, 1, 100)

    for (train_index, test_index) in cv.split(X, y):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        probs_rf = rf.fit(X_train, y_train).predict_proba(X_test)
        probs_abf = abf.fit(X_train, y_train).predict_proba(X_test)
        probs_xgb = xgb.fit(X_train, y_train).predict_proba(X_test)
        fpr_rf, tpr_rf, thresholds = roc_curve(y_test, probs_rf[:, 1])
        fpr_abf, tpr_abf, thresholds = roc_curve(y_test, probs_abf[:, 1])
        fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test, probs_xgb[:, 1])
        mean_tpr_rf += interp(mean_fpr_rf, fpr_rf, tpr_rf)
        mean_tpr_rf[0] = 0.0

        mean_tpr_abf += interp(mean_fpr_abf, fpr_abf, tpr_abf)
        mean_tpr_abf[0] = 0.0

        mean_tpr_xgb += interp(mean_fpr_xgb, fpr_xgb, tpr_xgb)
        mean_tpr_xgb[0] = 0.0

    mean_tpr_rf /= cv.get_n_splits(X, y)
    mean_tpr_rf[-1] = 1.0
    mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)

    mean_tpr_abf /= cv.get_n_splits(X, y)
    mean_tpr_abf[-1] = 1.0
    mean_auc_abf = auc(mean_fpr_abf, mean_tpr_abf)

    mean_tpr_xgb /= cv.get_n_splits(X, y)
    mean_tpr_xgb[-1] = 1.0
    mean_auc_xgb = auc(mean_fpr_xgb, mean_tpr_xgb)

    print('Mean auc random forest', mean_auc_rf)
    print('Mean auc Adaboost', mean_auc_abf)
    print('Mean auc XGBoost', mean_auc_xgb)

if __name__ == '__main__':
    main()