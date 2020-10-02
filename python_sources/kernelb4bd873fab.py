import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')

train_data = train_data[['Pclass','Sex','Age','Fare','Survived']]
test_data = test_data[['Pclass','Sex','Age','Fare']]

label_enc = LabelEncoder()
train_data['Sex'] = label_enc.fit_transform(train_data['Sex'])
test_data['Sex'] = label_enc.fit_transform(test_data['Sex'])

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

scaler = MinMaxScaler()
x_train = scaler.fit_transform(train_data.loc[:,['Pclass','Sex','Age']])
y_train = train_data['Survived']
x_test = scaler.fit_transform(test_data.loc[:,['Pclass','Sex','Age']])
y_test = gender_submission['Survived']

scaler_standard = StandardScaler()
scaler_standard.fit(train_data.loc[:,['Fare']])
x_train_fare = scaler_standard.transform(train_data.loc[:,['Fare']])
x_test_fare = scaler_standard.transform(test_data.loc[:,['Fare']])

x_train_all = np.concatenate([x_train,x_train_fare],axis=1)
x_test_all = np.concatenate([x_test,x_test_fare],axis=1)

gbc = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=6,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=5, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=200,
              presort='auto', random_state=555, subsample=1.0, verbose=0,
              warm_start=False)
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=8,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=7, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
            oob_score=False, random_state=555, verbose=0, warm_start=False)
svc = SVC(degree=4, gamma='auto', kernel='poly',C=100,probability=True)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=4, p=2,
           weights='uniform')
           
vot = VotingClassifier(
    estimators=[
        ('gbc',gbc),
        ('rfc',rfc),
        ('svc',svc),
        ('knn',knn),
    ],voting = 'soft'
)
vot.fit(x_train_all,y_train)
y_ = vot.predict(x_test_all)

submission = pd.DataFrame({'PassengerId':gender_submission['PassengerId'],'Survived':y_})
submission.to_csv('titanic.csv',index = False)


