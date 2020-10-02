# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib as plt
import seaborn as sn
from sklearn.metrics import r2_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
filename = os.listdir("../input")

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/'+filename[1])
data = pd.DataFrame(df)

features = data[["GRE Score","TOEFL Score", "University Rating", "SOP", "LOR ", "CGPA", "Research"]]
given_predictions = data[["Chance of Admit "]]

sn.pairplot(data, x_vars=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ','CGPA','Research'], y_vars='Chance of Admit ', height=7, aspect=0.7, kind='reg')
x_train, x_test, y_train, y_test = train_test_split(features, given_predictions, train_size=0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()
model = lm.fit(x_train, y_train)

print(lm.score(x_test, y_test))

weights = []
for i in range(len(features)):
    weights.append(model.coef_)

#print(weights)

y_predict = lm.predict(x_test)
print(r2_score(y_test, y_predict))