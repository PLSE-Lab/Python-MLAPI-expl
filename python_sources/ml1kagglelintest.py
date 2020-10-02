# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sklearn
#scikit-learn expects all data to be in numpy arrays of size [nsamples,nfeatures]

# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
x = np.array([0,1,2])
y = np.array([0,1,2])
X = x[:,np.newaxis] #input data is 2d and must be samplesXfeatures

model.fit(X,y)
model.coef_ #this prints out the weights
#model.predict(X_new) #outputs labels

