"""Super simple XGBoost model for the Kaggle Titanic dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xgb


def preproc(df):
    # -- Drop unpredictive attributes
    # Most likely: PassengerId, Ticket
    # Somewhat likely: Cabin, Name
    result = df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1)

    # -- Impute null values
    # Median: Age, Fare
    # Mode: Embarked (categorical)
    for col in ('Age', 'Fare'):
        result[col].fillna(df[col].median(), inplace=True)
    for col in ('Embarked', ):
        result[col].fillna(df[col].mode()[0], inplace=True)

    # -- Encode categorical values
    encoder = LabelEncoder()
    for col in ('Sex', 'Embarked'):
        result[col] = encoder.fit_transform(result[col])

    return result


def split_feats_target(df):
    X = df.drop('Survived', axis=1).values
    y = df['Survived'].values
    return X, y


def make_submission(filename, test_ids, test_preds):
    with open(filename, 'w') as outfile:
        outfile.write('PassengerId,Survived\n')
        for i, test_id in enumerate(test_ids):
            outfile.write('{},{}\n'.format(test_id, test_preds[i]))


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')

# Keep a copy of test passenger IDs for the final submission
test_ids = test_df['PassengerId'].values

train_df = preproc(train_df)
test_df = preproc(test_df)

train_corr = train_df.corr()
print(train_corr)

train_x, train_y = split_feats_target(train_df)
test_x = test_df.values

n_estims = 10
classif = xgb.sklearn.XGBClassifier(n_estimators=n_estims)
t0 = time.clock()
classif.fit(train_x, train_y)
t1 = time.clock()
print('Trained XGBoost classifier ({0} estimators) in {1:.2f} s'.format(
    n_estims, t1 - t0))

test_preds = classif.predict(test_x)
filename = 'submission.csv'
make_submission(filename, test_ids, test_preds)
print('Wrote results to {}'.format(filename))
