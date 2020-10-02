'''
import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
'''
import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
Y_train = train['Survived']
X_test = test[selected_features]


X_train['Embarked'].fillna('S', inplace = True)
X_test['Embarked'].fillna('S', inplace = True)

X_train['Age'].fillna(X_train['Age'].mean(), inplace = True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace = True)

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace = True)

#print(X_train.info())
#print(X_test.info())

from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse = False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = dict_vec.transform(X_test.to_dict(orient = 'record'))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


#XGBoost
from xgboost import XGBClassifier
xgbc = XGBClassifier()

#evaluation
from sklearn.cross_validation import cross_val_score
print(cross_val_score(rfc, X_train, Y_train, cv = 5).mean())
print(cross_val_score(rfc, X_train, Y_train, cv = 5).mean())

rfc.fit(X_train, Y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_y_predict})
rfc_submission.to_csv("rfc_submission.csv", index = False)

xgbc.fit(X_train, Y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':xgbc_y_predict})
xgbc_submission.to_csv("xgbc_submission.csv", index = False)

'''
from sklearn.grid_search import GridSearchCV
params = {'max_depth':list(range(2, 7)), "n_estimators":list(range(100, 1100, 200)),"learning_rate":[0.05,0.1,0.25,0.5,1.0]}


xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs = -1, cv = 5, verbose = 1)
gs.fit(X_train, Y_train)
print(gs.best_score_)
print(gs.best_params_)
xgbc_best_y_predict = xgbc_best.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':xgbc_best_y_predict})
xgbc_best_submission.to_csv("xgbc_best_submission.csv", xgbc_best_submission)
'''