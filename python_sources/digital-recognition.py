import pandas as pd
from pandas import DataFrame

from sklearn.ensemble import RandomForestClassifier
import numpy as np

import matplotlib.pyplot as plt

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

X_train, y_train = train_df.values[:, 1:], train_df.values[:, 0]
X_test = test_df.values

RF = RandomForestClassifier(n_estimators= 1000, n_jobs= 6)
RF.fit(X_train, y_train)

RF.score(X_train, y_train)