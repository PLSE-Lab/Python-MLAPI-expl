import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("../input/heart.csv")
#print(data)
name=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
X=data[name]
#print(X)
Y=data['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)
m=DecisionTreeClassifier()
m.fit(X_train,Y_train)
print(m.score(X_test,Y_test)*100)
y_pred=m.predict([[41,0,1,130,204,0,0,172,0,1.4,2,0,2],
                 [60,1,0,117,230,1,1,160,1,1.4,2,2,3],
                  [32,1,1,110,210,0,1,170,0,0,2,0,3]])
print(y_pred)