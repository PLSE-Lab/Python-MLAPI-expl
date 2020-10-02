import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import lightgbm as lgb

train = pd.read_csv('../input/m_train_small.csv')
test = pd.read_csv('../input/m_test_small.csv')

train_labels = train['TARGET']
train_ids = train['SK_ID_CURR']
test_ids = test['SK_ID_CURR']

train = train.drop(columns = ['TARGET', 'SK_ID_CURR'])
test = test.drop(columns = ['SK_ID_CURR'])

submission = pd.DataFrame({'SK_ID_CURR' : test_ids})

pipeline = Pipeline([('imputer', Imputer(strategy = 'median')),
                     ('scaler', MinMaxScaler(feature_range = (0, 1)))])
                     
train = pipeline.fit_transform(train)
test = pipeline.transform(test)

print('Training shape: ', train.shape)
print('Testing shape: ', test.shape)

def make_submission(model, name):
    predictions = model.predict_proba(test)[:, 1]
    submission['TARGET'] = predictions
    submission.to_csv('%s_submission.csv' % name, index = False)
    print('Submission saved to %s_submission.csv' % name)
    
logreg = LogisticRegressionCV(Cs = 20, cv = 3, verbose = 1)
logreg.fit(train, train_labels)
make_submission(logreg, name = 'logreg')

rf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1, verbose = 1)
rf.fit(train, train_labels)
make_submission(rf, name = 'rf')

et = ExtraTreesClassifier(n_estimators = 1000, n_jobs = -1, verbose = 1)
et.fit(train, train_labels)
make_submission(et, name = 'et')

gbm = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.01, verbose = 1)
gbm.fit(train, train_labels)
make_submission(gbm, name = 'gbm')

# Create the model with several hyperparameters
lgb_gbm = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 1000, 
                             learning_rate = 0.01, class_weight = 'balanced', n_jobs = -1, verbose = 200)
lgb_gbm.fit(train, train_labels)
make_submission(lgb_gbm, 'lgb_gbm')

# Read in the submissions 
logreg_sub = pd.read_csv('logreg_submission.csv')
rf_sub = pd.read_csv('rf_submission.csv')
et_sub = pd.read_csv('et_submission.csv')
gbm_sub = pd.read_csv('gbm_submission.csv')
lgb_gbm_sub = pd.read_csv('lgb_gbm_submission.csv')

average_sub = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': 0})

# Average the preditions together
average_sub['TARGET'] = (rf_sub['TARGET'] + et_sub['TARGET'] + gbm_sub['TARGET'] + lgb_gbm_sub['TARGET']) / 4
average_sub.to_csv('average_sub.csv', index = False)