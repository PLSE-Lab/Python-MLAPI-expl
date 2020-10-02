# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/Salaries.csv')

data = data.sample(frac=1).reset_index(drop=True)

v = int((4.0/8.0)*len(data))
t = int((6.0/8.0)*len(data))

xColumns = ["JobTitle","Year","TotalPay"]
yColumns = ["TotalPayBenefits"]

le = LabelEncoder()
le.fit(data["JobTitle"])
data["JobTitle"] = le.transform(data["JobTitle"])

train = data[:v]
validation = data[v:t]
test = data[t:]

trainX = train[xColumns]
trainY = train[yColumns]
validationX = validation[xColumns]
validationY = validation[yColumns]
testX = test[xColumns]
testY = test[yColumns]

# trainX = np.asarray(trainX, dtype="float64")
# trainY = np.asarray(trainY, dtype="float64")

# print(trainX.head(),trainY.head())

gb = GradientBoostingRegressor()
gb.fit(trainX,trainY)
# rf.fit(trainX,trainY)

preds = gb.predict(validationX)

print(gb.score(trainX,trainY))
print(gb.score(validationX,validationY))

validationY["Predicted"] = preds
# print(preds)

validationY["JobTitle"] = validationX["JobTitle"]
validationY["JobTitle"] = le.inverse_transform(validationY["JobTitle"])
validationY["Year"] = validationX["Year"]
validationY["TotalPay"] = validationX["TotalPay"]
validationY.sort_values("TotalPay",inplace=True)
print(validationY.tail(100))




# print(train.head(100))

# Any results you write to the current directory are saved as output.