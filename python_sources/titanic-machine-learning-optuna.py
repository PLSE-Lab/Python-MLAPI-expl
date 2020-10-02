# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import scipy
from sklearn.linear_model import LogisticRegression
import optuna

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train=pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')
test=pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')
submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')

print("Training Data: {} & Test Data: {}".format(train.shape, test.shape))
Ytrain=train['Survived'] # setting target
train=train[list(test)]
total_data=pd.concat((train, test))  # all training + test data set
print(train.shape, test.shape, total_data.shape)

# Now encode the columns

encoded=pd.get_dummies(total_data, columns=total_data.columns, sparse=True)
encoded=encoded.sparse.to_coo()
encoded=encoded.tocsr()

Xtrain=encoded[:len(train)]
Xtest=encoded[len(train):]

kf=StratifiedKFold(n_splits=10)


"""
Test to find best value for C
In this case C=9.89333631520736

Finished trial#49 resulted in value: -0.8550657555951673. Current best value is -0.874178055060408 with parameters: {'C': 9.89333631520736}.

def test(trial):
    C=trial.suggest_loguniform('C', 10e-10, 10)
    model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)
    score=-cross_val_score(model, Xtrain, Ytrain, cv=kf, scoring='roc_auc').mean()
    return score
t=optuna.create_study()


t.optimize(test, n_trials=50)

print(t.best_params)

#print(-study.best_value)
#params=study.best_params
"""

## Model Preration

model=LogisticRegression(C=9.89333631520736, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)
model.fit(Xtrain, Ytrain)
#predictions=model.predict_proba(Xtest)[:,1] # probability of getting 1 replace '1' with 0 to get probability of 0
predictions=model.predict(Xtest) # to get target column as either 0 or 1 value no probability
submission['Survived']=predictions
submission.to_csv('submit.csv')
submission.head()










