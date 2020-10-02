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

import pandas as pd

path = "../input/cereal.csv"
cerealDF = pd.read_csv(path)


X = cerealDF.drop(['name', 'mfr', 'type'], axis=1)
y = cerealDF.rating

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
cereal_model = RandomForestRegressor()
cereal_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error
predicted_ratings = cereal_model.predict(X_test)

print(mean_absolute_error(y_test, predicted_ratings))