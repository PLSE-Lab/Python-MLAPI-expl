# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from rgf import rgf
from sklearn.cross_validation import StratifiedKFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

iris = pd.read_csv('../input/Iris.csv')
iris_data = iris.loc[ 1: , ['SepalLengthCm', 'SepalWidthCm'  ,'PetalLengthCm','PetalWidthCm']]
iris_data = np.array(iris_data)


iris['Target_Species'] = iris.Species.map({"Iris-setosa": 0,
                                           "Iris-versicolor": 1,
                                           "Iris-virginica": 2})
iris_target = iris.loc[ 1: , ['Target_Species']]
iris_target = np.array(iris_target)
iris_target = np.ravel(iris_target)

model = rgf.RGFClassifier(max_leaf=500,
                          algorithm="RGF_Sib",
                          test_interval=100,)
                          

# cross validation
score = 0
n_folds = 3

for train_idx, test_idx in StratifiedKFold(iris_target, n_folds):
    xs_train = iris_data[train_idx]
    y_train = iris_target[train_idx]
    xs_test = iris_data[test_idx]
    y_test = iris_target[test_idx]

    model.fit(xs_train, y_train)
    score += model.score(xs_test, y_test)

score /= n_folds
print('score: {0}'.format(score))

for i in range(1,100000):
    n = i / 99
    print(n);
