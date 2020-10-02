import numpy as np
import pandas as pd
import pylab as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv('../input/data.csv',header=0)

X_train, X_test, y_train, y_test = train_test_split(df.drop('is_weekend', axis=1), df['is_weekend'], test_size=0.3)

reg = LogisticRegression()
y_pred = reg.fit(X_train, y_train).predict(X_test)
print("Score of logistic regression",reg.score(X_test,y_test))
print("Prediction matrix of logistic regression")
print(reg.predict(X_test))


radm = RandomForestClassifier()
radm.fit(X_train,y_train)
print("Score of Random forest",radm.score(X_test,y_test))
print("Prediction matrix of Random forest")
print(radm.predict(X_test))


svm = SVC()
svm.fit(X_train,y_train)
print("Score of SVM",svm.score(X_test,y_test))
print("Prediction matrix of SVM")
print(svm.predict(X_test))