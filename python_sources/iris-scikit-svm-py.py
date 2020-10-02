# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import csv
import numpy as np
from sklearn import svm
import random

iris = []
lab2ind = {}
ind2lab = []
with open('../input/Iris.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        iris.append(row)
        if row[-1] not in lab2ind:
            lab2ind[row[-1]] = len(lab2ind)
            ind2lab.append(row[-1])
random.shuffle(iris)
x = np.array([[float(num) for num in row[1:-1]] for row in iris])
y = np.array([lab2ind[row[-1]] for row in iris])
train_x = x[:120]
test_x = x[120:]
train_y = y[:120]
test_y = y[120:]
clf = svm.SVC()
clf.fit(train_x,train_y)
result = clf.predict(test_x)
print('accuracy',np.sum(result == test_y)/float(len(test_x)))