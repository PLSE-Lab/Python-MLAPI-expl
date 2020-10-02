#Import statements
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression

#Read CSV File
df=pd.read_csv('../input/HR_comma_sep.csv',header=0)

#Map department to numeric values
df['department']=df['sales'].map({'sales':0, 'accounting':1, 'hr':2, 'technical':3, 'support':4, 'management':5, 'IT':6,
 'product_mng':7, 'marketing':8, 'RandD':9}).astype(int)

#Map salary to numeric values
df['pay']=df['salary'].map({'low':0, 'medium':1, 'high':2}).astype(int)

#Drop columns which have text values
df=df.drop(['sales','salary'],axis=1)

#Drop null rows if any
df=df.dropna()

#Convert float64 values to int64 values
df['satisfaction_level'] = (df['satisfaction_level'] * 100).astype(int)
df['last_evaluation'] = (df['last_evaluation'] * 100).astype(int)

#Split the data into Train and Test data
X_train, X_test, y_train, y_test = train_test_split(df.drop('left', axis=1), df['left'], test_size=0.3)

#Apply Logistic Regression Algorithm
LR = LogisticRegression()
prediction = LR.fit(X_train, y_train).predict(X_test)
print(prediction)
print("Logistic Regression Score", LR.score(X_test,y_test)) #0.797111111111

#Apply Bagging Regressor Algorithm
BG=BaggingRegressor(n_estimators=100)
prediction=BG.fit(X_train, y_train).predict(X_test)
print(prediction)
print("Bagging Regression Score", BG.score(X_test,y_test)) #0.943849374781

#Apply Random Forest Classifier
RF=RandomForestClassifier(n_estimators=100)
prediction=RF.fit(X_train, y_train).predict(X_test)
print(prediction)
print("Forest Classifier Score", RF.score(X_test,y_test)) #0.992666666667
