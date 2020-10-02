# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def data_cleanup(titanic_data):
    sex_dummy = pd.get_dummies(titanic_data.Sex)
    titanic_data['male'] = sex_dummy.male
    titanic_data['female'] = sex_dummy.female
    
    titanic_data.Age = titanic_data.Age.fillna(titanic_data.Age.median())
    titanic_data.Fare = titanic_data.Fare.fillna(titanic_data.Fare.median())
    
    return titanic_data
# X_features = train_data[['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']].copy()

train = data_cleanup(train)
test = data_cleanup(test)

X_features_train = train[['Age','SibSp','Parch','Fare']].copy()
X_features_test  = test[['Age','SibSp','Parch','Fare']].copy()
print(X_features_test.info)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

rf = RandomForestClassifier(n_estimators=50,max_depth=20,n_jobs=-1)
rf_model = rf.fit(X_features_train,train['Survived'])
y_pred = rf_model.predict(X_features_test)
# print(np.size(y_pred))
# print(y_pred)
# print(np.shape(X_features_test))
res = pd.DataFrame(y_pred,index=test['PassengerId'])
res.columns = ['Survived']
print(res.info)
res.to_csv('result')
# print(res.head())
# k_fold = KFold(n_splits = 5)
# cross_val_score(rf,X_features,train['Survived'],cv = k_fold,scoring='accuracy')
























