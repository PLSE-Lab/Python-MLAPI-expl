# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 987

def make_submit(y_pred):
    submit = pd.read_csv('../input/TechJam2019/sample_submission_v1.csv')
    submit.iloc[:, 1:] = y_pred
    return submit



def day2month(x):
    days = np.array([31,29,31,30,30,30,31,31,30,31,30,31]).cumsum()
    month = np.arange(1, 13)

    for n, d in zip(month, days):
        if x<=d:
            return n
        
def get_data():
    train = pd.read_csv('../input/TechJam2019/train.csv')
    test = pd.read_csv('../input/TechJam2019/test.csv')
    
    # demo
    print("\t-preparing: demo.csv")
    demo = pd.read_csv('../input/TechJam2019/demo.csv')
    demo['c1'].fillna(demo.c1.mode()[0], inplace=True)
    demo.fillna(0, inplace=True)
    demo['n1_plus_n2'] = demo['n1'] + demo['n2']
    demo['n2_minus_n1'] = demo['n2'] - demo['n1']

    le = LabelEncoder()
    cat_cols = ['c0', 'c1', 'c2', 'c3', 'c4']
    for c in cat_cols:
        demo[c] = le.fit_transform(demo[c])
    train = pd.merge(train, demo, on='id')
    test = pd.merge(test, demo, on='id')

    # prep txn
    print('\t-preparing: txn.csv')
    txn = pd.read_csv('../input/TechJam2019/txn.csv')
    txn['day_of_week'] = txn['n3'] % 7
    txn['month'] = txn['day_of_week'].map(day2month)
    
    n4_by_id = txn.groupby('id').n4.agg(['min', 'max', 'mean', 'median', 'std', 'count'])
    n4_by_id.columns = 'n4_' + n4_by_id.columns

    n4_by_id_month = txn.groupby(['id', 'month']).n4.agg(['min', 'max', 'mean', 
                                                          'median', 'std', 'count']).unstack().fillna(0)

    n4_by_id_month.columns = [n4_by_id_month.columns.names[1]+'_'+a+"_"+str(b) for a,b in n4_by_id_month.columns]

    n4_by_id_dow = txn.groupby(['id', 'day_of_week']).n4.agg(['min', 'max', 'count', 
                                                              'mean', 'median', 'std']).unstack().fillna(0)
    n4_by_id_dow.columns = [n4_by_id_dow.columns.names[1]+'_'+a+"_"+str(b) for a,b in n4_by_id_dow.columns]
    no_cc_by_id = txn.groupby('id').old_cc_no.nunique()

    c5_count_by_id = txn.groupby('id').c5.value_counts(normalize=True).unstack().fillna(0)
    c5_count_by_id.columns = c5_count_by_id.columns.name + '_' + c5_count_by_id.columns.astype(str)

    txn_by_id = pd.concat([n4_by_id, n4_by_id_month, n4_by_id_dow, no_cc_by_id, c5_count_by_id], axis=1)


    train = pd.merge(train, txn_by_id, left_on='id', right_index=True, how='left')
    test = pd.merge(test, txn_by_id, left_on='id', right_index=True, how='left')

    y = train.pop('label')
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    
    return train, y, test

print("Loading data...")
train, y, test = get_data()
X_train, X_val, y_train, y_val = train_test_split(train, y, stratify=y, test_size=.25, random_state=SEED)

cat_cols = ['c0', 'c1', 'c2', 'c3', 'c4']
dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
dval = lgb.Dataset(X_val, y_val, reference=dtrain)

init_params = {'boosting_type': 'gbdt',
             #'class_weight': 'balanced',
             'colsample_bytree': 1.0,
             'importance_type': 'split',
             'learning_rate': 0.05,
             'max_depth': -1,
             'min_child_samples': 20,
             'min_child_weight': 0.001,
             'min_split_gain': 0.0,
             'n_jobs': 2,
             'num_leaves': 31,
            'objective': 'multiclass',
            'num_class':13,
             'random_state': SEED,
             'reg_alpha': 0.0,
             'reg_lambda': 0.0,
             'subsample': 1.0,
             'subsample_for_bin': 200000,
             'subsample_freq': 0}
print("Training model...")
m_lgb = lgb.train(init_params, dtrain, num_boost_round=2000, valid_sets=dval,
              valid_names='val', early_stopping_rounds=30)

print("Making prediction")
y_pred = m_lgb.predict(test, num_iteration=m_lgb.best_iteration)
submit = make_submit(y_pred)
submit.to_csv('O_0466.csv', index=False)

print("Submission done")