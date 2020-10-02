# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
Nycab_file_path = '/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'
Nycab_data = pd.read_csv(Nycab_file_path)
Nycab_data.describe()
Nycab_data.columns
y = Nycab_data.price
Nycab_features = ['number_of_reviews', 'availability_365', 'longitude', 'latitude']
X = Nycab_data[Nycab_features]
X.describe()

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)
from sklearn.tree import DecisionTreeRegressor
Nycab_model= DecisionTreeRegressor()
Nycab_model.fit(train_X,train_y)
val_predictions = Nycab_model.predict(val_X)
print(val_predictions)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(val_y,val_predictions))
print(val_y)