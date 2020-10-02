# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/Interview.csv')

#drop unnecessary columns

data = data.drop(list(data.columns)[-5:], axis=1)

#drop nan

data = data.dropna()

#data['Date of Interview'] = data['Date of Interview'].map(lambda x : x.split('&'))

#drop unnecessary columns
data = data.drop(['Name(Cand ID)', 'Date of Interview'], axis=1)


keys = list(data.columns)
values = [pd.factorize(data[x])[0] for x in keys]

data = pd.DataFrame({k:v for k,v in zip(keys,values)})

x = [x for x in keys if x != 'Observed Attendance']
x = data[x]

print(list(x.columns))

y = data['Observed Attendance']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3)

imp_features = ExtraTreesClassifier().fit(x_train, y_train).feature_importances_

imp_features = {k : v for k,v in zip(keys, imp_features)}

imp_features = Counter(imp_features).most_common(5)

imp_features = [name[0] for name in imp_features]

#fit and train 



features = preprocessing.normalize(data[imp_features])
x_train, x_test, y_train, y_test = train_test_split(features,y, test_size=.3)

clf = SVC(kernel='rbf', C=50).fit(x_train, y_train)
pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)

print('SVM accracy is ', accuracy)


xclf = XGBClassifier().fit(x_train, y_train)
pred = xclf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print('XGBOOST accuracy ', accuracy)



bclf = BaggingClassifier().fit(x_train, y_train)
pred = bclf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print('Bagging score : ' , accuracy)


#kmean elbow method


clusters = [num for num in range(1,21)]
errors = []

for num_clusters in clusters:
    
    clf = KMeans(n_clusters = num_clusters).fit(x_train)
    errors.append(clf.inertia_)
    
    
optimal_clusters = {k : v for k,v in zip(clusters, errors)}

plt.plot(clusters, errors, marker='o')
plt.title('ELBOW METHOD to determine optimal number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Errors')
plt.show()


clf = LinearSVC().fit(x_train, y_train)
pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print('linear support vector machine : ', accuracy)