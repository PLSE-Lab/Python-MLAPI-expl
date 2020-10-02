import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
selected_features=['Pclass','Sex','Age','Embarked','SibSp','Parch','Fare']
X_train=train[selected_features]
X_test=test[selected_features]
y_train=train['Survived']

#Print to standard output, and see the results in the "log" section below after running your script
#print(train.info())
#print(test.info())
#print(X_train['Embarked'].value_counts())
#print(X_test['Embarked'].value_counts())
X_train['Embarked'].fillna('S',inplace=True)
X_test['Embarked'].fillna('S',inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)
#print(X_train.info())
#print(X_test.info())

from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
print(X_train)
X_test=dict_vec.fit_transform(X_test.to_dict(orient='record'))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
from xgboost import XGBClassifier
xgbc=XGBClassifier()
from sklearn.cross_validation import cross_val_score
cross_val_score(rfc,X_train,y_train,cv=5).mean()
cross_val_score(xgbc,X_train,y_train,cv=5).mean()
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)
rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})
xgbc.fit(X_train,y_train)
xgbc_y_predict=xgbc.predict(X_test)
xgbc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgbc_y_predict})


#Any files you save will be available in the output tab below
rfc_submission.to_csv('rfc_submission.csv', index=False)
xgbc_submission.to_csv('xgbc_submission.csv', index=False)

