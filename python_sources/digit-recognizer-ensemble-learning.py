# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

new_data = train_data.drop(columns=['label'])

train_label = train_data['label']

Pred1 = pd.DataFrame()

ada = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=400, max_depth=10, bootstrap=True),
                         n_estimators=500, algorithm='SAMME', learning_rate=0.1)

svc = SVC(kernel='poly', C=10, gamma=0.1)

parameters = {'learning_rate': [0.01],
              'n_estimators': [500, 1000],
              'gamma': [0.0001],
              'max_depth': [12, 20]}

x = xgboost.XGBClassifier()

vc = VotingClassifier([('ada', ada), ('svc', svc), ('xgb', x)])

# km = KMeans(max_iter=800, n_clusters=10, n_init=50, random_state=2020)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.1)

val_fit = ada.fit(X_train, Y_train)
val_pred = ada.predict(X_test)
#
val_fit_svc = svc.fit(X_train, Y_train)
val_pred_svc = svc.predict(X_test)
#
# k_fit = km.fit_predict(train_data)

# Error = []
# for i in range(1, 20):
#     kmeans = KMeans(n_clusters=i).fit(train_data)
#     kmeans.fit(train_data)
#     Error.append(kmeans.inertia_)
#
# plt.plot(range(1, 20), Error)
# plt.title('Elbow method')
# plt.xlabel('No of clusters')
# plt.ylabel('Error')
# plt.show()
#
# plt.scatter(train_data[:, 0], train_data[:, 1], c=k_fit, cmap='rainbow')
# plt.show()

x_search = GridSearchCV(estimator=x, param_grid=parameters, verbose=3, cv=5)

mlp_train = x_search.fit(X_train, Y_train)

mlp_prediction = x_search.predict(X_test)

vc_fit = vc.fit(X_train, Y_train)
vc_pred = vc.predict(X_test)

print("Accuracy of the AdaBoost model is ", accuracy_score(Y_test, val_pred))
print("Accuracy of the SVC model is ", accuracy_score(Y_test, val_pred_svc))
print("Accuracy of the XGB model is ", accuracy_score(Y_test, mlp_prediction))
print("Accuracy of the VC model is ", accuracy_score(Y_test, vc_pred))

fitted = x_search.best_estimator_.fit(new_data, train_label)
pred = fitted.predict(test_data)

test_data['label'] = pred
train_data['label'] = train_label

test_data['label'].to_csv('Label.csv', index=True, index_label='ImageId')

complete_data = pd.concat([train_data, test_data], ignore_index=True, sort=True)

complete_data_label = complete_data['label']

complete_data.drop(columns=['label'], inplace=True)

xgb_fit = x_search.best_estimator_.fit(complete_data, complete_data_label)

xgb_predict = xgb_fit.predict(test_data)

test_data['label'] = xgb_predict

test_data['label'].to_csv('Result.csv', index=True, index_label='ImageId')
