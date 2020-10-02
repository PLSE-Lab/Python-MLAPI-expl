# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# %% [code]
data = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

data = data.drop(['Serial No.'], axis = 1)
features = data.drop(['Chance of Admit '], axis = 1)
labels = data['Chance of Admit ']



train_feats = features[:400]
train_labels = labels[:400]
test_feats = features[400:]
test_labels = labels[400:]

scale = StandardScaler()
train_feats = scale.fit_transform(train_feats)
test_feats = scale.transform(test_feats)


model = LinearRegression()
model.fit(train_feats, train_labels)
predictions = model.predict(test_feats)
error = np.sqrt(mean_squared_error(test_labels, predictions))
                        
submission = pd.DataFrame([error], columns = ['RMSE score'])
submission.to_csv("submission.csv",index=False)