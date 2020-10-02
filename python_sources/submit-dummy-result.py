# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.dummy import DummyRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# load data
df_train = pd.read_csv("../input/traindata.csv")
df_test = pd.read_csv("../input/testdata.csv")
y_train = pd.read_csv("../input/traindata_label.csv").values

# Just take a subset of the available features
x_train = df_train[["4046", "4225", "4770", "Total Bags", "Total Volume"]].values
x_test = df_test[["4046", "4225", "4770", "Total Bags", "Total Volume"]].values

model = DummyRegressor()
model.fit(x_train, y_train)

# Any results you write to the current directory are saved as output.
train_pred = model.predict(x_train)
test_pred = pd.DataFrame(model.predict(x_test))
test_pred.to_csv("dummy_submitted_from_kernel.csv", header=["AveragePrice"], index_label="ID")
