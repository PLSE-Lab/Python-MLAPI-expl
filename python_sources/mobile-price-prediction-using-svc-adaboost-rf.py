# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('../input/mobile-price-classification/train.csv')
X = df.drop('price_range',axis=1)
y = df['price_range']

print('\n Performing Standardization')

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)

print('\n Done')

print('\n Train Test Split')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print('\n Done')

print('\n Model building')

print('\n Building Support Vector Classifier')

from sklearn.svm import SVC
svc = SVC()
model_svc = svc.fit(X_train,y_train)
y_pred_svc = model_svc.predict(X_test)
from sklearn.metrics import classification_report
print('\n Done')
print('\n <======= Accuracy Report =======>')
print(classification_report(y_test,y_pred_svc))

print('\n Building Random Forest')

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,criterion='entropy')
model_rf = rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)
print('\n Done')
print('\n <======= Accuracy Report =======>')
print(classification_report(y_test,y_pred_rf))

print('\n Building AdaBoostClassifier')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ad = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',splitter='best'))
model_ad = ad.fit(X_train,y_train)
y_pred_ad = model_ad.predict(X_test)
print('\n Done')
print('\n <======= Accuracy Report =======>')
print(classification_report(y_test,y_pred_ad))




# Any results you write to the current directory are saved as output.