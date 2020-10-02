# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv').as_matrix()
test = pd.read_csv('../input/test.csv').as_matrix()


X_train = train[:,:562]
y_train = train[:,562]

print(X_train.shape, y_train.shape)

print(y_train[0])