import pandas as pd
import numpy as np

import category_encoders as ce

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

data = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
prev = pd.read_csv('../input/previous_application.csv')
buro = pd.read_csv('../input/bureau.csv')

submit = test[['SK_ID_CURR']]


def categorical_features(df):
    return [f for f in df.columns if df[f].dtype == 'object']
    

y = data['TARGET']
data.drop(['TARGET'], axis=1, inplace=True)

cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(cnt_prev['SK_ID_PREV'])

avg_prev = prev.groupby('SK_ID_CURR').mean()
avg_prev.columns = ['prev_app_' + f_ for f_ in avg_prev.columns]

avg_buro = buro.groupby('SK_ID_CURR').mean()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']

avg_buro.columns = ['bureau_' + f_ for f_ in avg_buro.columns]

# merge all the data
data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

cat_features = categorical_features(data)
enc = ce.backward_difference.BackwardDifferenceEncoder(cols=cat_features).fit(data, y)

data = enc.transform(data)
test = enc.transform(test)

pd.set_option('max_columns', None)
data.describe()

# display(data.columns)
# display(test.columns)
np.setdiff1d(data.columns, test.columns)
# data[np.setdiff1d(data.columns, test.columns)]
data.drop(np.setdiff1d(data.columns, test.columns), axis=1, inplace=True)

# Get features
excluded_feats = ['SK_ID_CURR']
features = [f_ for f_ in data.columns if f_ not in excluded_feats]

# Run a 5 fold
folds = KFold(n_splits=5, shuffle=True, random_state=42)
val_preds = np.zeros(data.shape[0])
test_preds = np.zeros(test.shape[0])

for n_fold, (train_idx, val_idx) in enumerate(folds.split(data)):
    train_x, train_y = data[features].iloc[train_idx], y.iloc[train_idx]
    val_x, val_y = data[features].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        silent=True,
        verbose=-1,
    )
    
    clf.fit(train_x, train_y,
            eval_set=[(train_x, train_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=150
           )
    
    val_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    test_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold {:d} AUC: {:.6f}'.format(n_fold + 1, roc_auc_score(val_y, val_preds[val_idx])))

print('Full AUC score: {:.6f}'.format(roc_auc_score(y, val_preds)))
submit['TARGET'] = test_preds

submit[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False, float_format='%.8f')