# handling
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# model
import lightgbm as lgb

# unsupervised learning
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# prevent overfit
from sklearn.model_selection import KFold

# warning
import warnings
warnings.filterwarnings('ignore')


tr = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')
te = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')
sub = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/sample_submission.csv')

price = np.log1p(tr['price'])

oof_stack = pd.read_csv('../input/hhl-8-models/oof_results.csv')
pred_stack = pd.read_csv('../input/hhl-8-models/pred_results.csv')

features = oof_stack.columns
scaler = StandardScaler()
oof_stack = round(pd.DataFrame(scaler.fit_transform(oof_stack), columns=features), 2)
pred_stack = round(pd.DataFrame(scaler.fit_transform(pred_stack), columns=features), 2)

kmeans_df = pd.concat([oof_stack, pred_stack])
kmeans = KMeans(n_clusters=200, random_state=1028).fit(kmeans_df[features])
kmeans_df['kmeans_labels_1'] = kmeans.labels_

kmeans = KMeans(n_clusters=50, random_state=1028).fit(kmeans_df[features])
kmeans_df['kmeans_labels_2'] = kmeans.labels_

DBscan = DBSCAN(eps=0.3, min_samples=2).fit(kmeans_df[features])
kmeans_df['DBSCAN_labels_1'] = DBscan.labels_

kmeans_df = kmeans_df.reset_index(drop=True)
oof_stack = kmeans_df.loc[:oof_stack.index[-1]]
pred_stack = kmeans_df.loc[oof_stack.index[-1]+1:].reset_index(drop=True)


pred_stack['skew'] = pred_stack.skew(1)
pred_stack['kurt'] = pred_stack.kurt(1)

oof_stack['skew'] = oof_stack.skew(1)
oof_stack['kurt'] = oof_stack.kurt(1)

random_state=1028
param = {
    "objective" : "regression",
    "metric" : "rmse",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 5,
    "min_data_in_leaf": 5,
    "bagging_freq": 5,
    "learning_rate" : 0.01,
    "bagging_fraction" : 0.464,
    "feature_fraction" : 0.582,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "verbosity" : 1,
    "feature_fraction_seed" : random_state,
    "bagging_fraction_seed" : random_state,
    "random_state": random_state
}   
splits = 5
kf = KFold(n_splits=splits, shuffle=True, random_state=1028)

oof = np.zeros(len(tr))
predictions = np.zeros(len(te))

for trn_idx, val_idx in kf.split(tr, price):
    
    train_X, train_y = oof_stack.loc[trn_idx], price[trn_idx]
    test_X, test_y = oof_stack.loc[val_idx], price[val_idx]
    
    lgb_train = lgb.Dataset(train_X, label=train_y)
    lgb_valid = lgb.Dataset(test_X, label=test_y)

    clf = lgb.train(param, lgb_train, 100000, valid_sets = [lgb_train, lgb_valid], early_stopping_rounds = 500, verbose_eval=10000)

    oof[val_idx] = clf.predict(test_X, num_iteration=clf.best_iteration)
    predictions += clf.predict(pred_stack, num_iteration=clf.best_iteration)/splits
    

sub['price'] = np.exp(predictions*1.0009)
sub.to_csv('final.csv', index=False)