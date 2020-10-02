# This Starter Kernel should beat the Random Forest Benchmark and serve as a basis for more advanced models.

import numpy as np 
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

# get data and splits
train = pd.read_csv('../input/train.csv')
trainX = train[train.columns[2:]].values
trainy = train['target'].values

# run regression
etr = ExtraTreesRegressor(min_samples_leaf=2, verbose=1, n_jobs=-1)
etr = etr.fit(trainX, np.log1p(trainy))

#predict
test = pd.read_csv('../input/test.csv')
testX = test[test.columns[1:]].values
t_preds = etr.predict(testX)

#submit
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = np.around(np.expm1(t_preds), 0)
sub.to_csv('sub_et.csv', index=False)