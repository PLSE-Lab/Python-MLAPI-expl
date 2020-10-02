import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_dt = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
test_dt = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
submission_dt = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

ids = submission_dt['ForecastId']

input_cols = ["Lat","Long","Date"]
output_cols = ["ConfirmedCases","Fatalities"]

for i in range(train_dt.shape[0]):
    train_dt["Date"][i] = train_dt["Date"][i][:4] + train_dt["Date"][i][5:7] + train_dt["Date"][i][8:]
    train_dt["Date"][i] = int(train_dt["Date"][i])
	
for i in range(test_dt.shape[0]):
    test_dt["Date"][i] = test_dt["Date"][i][:4] + test_dt["Date"][i][5:7] + test_dt["Date"][i][8:]
    test_dt["Date"][i] = int(test_dt["Date"][i])
	
X = train_dt[input_cols]
Y1 = train_dt[output_cols[0]]
Y2 = train_dt[output_cols[1]]

X_test = test_dt[input_cols]

sk_tree = DecisionTreeClassifier(criterion='entropy')

sk_tree.fit(X,Y1)

pred1 = sk_tree.predict(X_test)

sk_tree.fit(X,Y2)

pred2 = sk_tree.predict(X_test)

ids.shape

pred1.shape

pred2.shape

output = pd.DataFrame({ 'ForecastId' : ids, 'ConfirmedCases': pred1,'Fatalities':pred2 })
output.to_csv('submission.csv', index=False)