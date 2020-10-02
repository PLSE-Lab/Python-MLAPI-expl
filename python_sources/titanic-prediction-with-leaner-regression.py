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
from sklearn.cross_validation import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics     

df = pd.read_csv("../input/train.csv")
to_drop = ["PassengerId","Name","Ticket","Cabin"]
df.drop(to_drop,axis = 1,inplace = True)
df["Num_Sex"] = df["Sex"].map({"male":0,"female":1})
df["Num_Embarked"] = df["Embarked"].map({"C":0,"S":1,"Q":2})
df = df.dropna()
x_col = ["Pclass","Age","SibSp","Fare","Num_Embarked","Num_Sex"]
feature_col = ["Pclass","Age","SibSp","Fare","Num_Sex"]
x = df[feature_col]
y = df.Survived
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 1)
linreg = LinearRegression()
linreg.fit(x_train,y_train)
zip(feature_col, linreg.coef_)
y_pred = linreg.predict(x_test)
y_pred = np.where(y_pred > 0.5,1,0)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
