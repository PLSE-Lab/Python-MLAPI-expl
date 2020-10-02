# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('./../input/voice.csv')

print(df.head(3))

le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

clf = XGBClassifier()
kfold =  KFold(len(y), n_folds=10)
results = cross_val_score(clf, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

