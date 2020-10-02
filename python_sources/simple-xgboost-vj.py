import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=42,random_state=42)

model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
model.fit(x_train,y_train)
pred = model.predict(x_test) 

print(confusion_matrix(pred,y_test))
print(accuracy_score(pred,y_test))
#predictions = gbm.predict()