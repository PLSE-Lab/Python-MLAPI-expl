# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('/kaggle/input/Social_Network_Ads.csv')
X= dataset.iloc[:,[2,3]].values
y= dataset.iloc[:,4].values

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Fitting Random Forest Classification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=50)
classifier.fit(X_train, y_train)

# Predicting the result
y_pred = classifier.predict(X_test)

# Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
acc = accuracy_score(y_test, y_pred)
acc*=100
print("Accuracy before RandomizedSearchCV: {} %".format(acc))
print(classification_report(y_test, y_pred))


# Function for RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
def hypertuning_rscv(est, p_dist, nbr_iter, X, y):
    rand_search_cv = RandomizedSearchCV(est, param_distributions=p_dist, n_jobs=-1, n_iter=nbr_iter, cv=9)
    rand_search_cv.fit(X,y)
    ht_params = rand_search_cv.best_params_
    ht_score = rand_search_cv.best_score_
    return ht_params, ht_score

# Funtion Call Preparation
est = RandomForestClassifier(n_jobs=-1)
p_dist = {
    'max_depth':[3,5,10,None],
    'max_features':randint(1,3),
    'n_estimators':[10, 100, 200, 300, 400, 500],
    'criterion':['gini','entropy'],
    'bootstrap':[True, False],
    'min_samples_leaf':randint(1,4),
}

# Funtion Call
rf_params, rf_ht_score = hypertuning_rscv(est, p_dist, 40, X, y)
print("Best Parameters:{}".format(rf_params))
print("Hyper Tuning Score:{}".format(rf_ht_score))

# Applying Best Parameters
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=5, max_features=1, min_samples_leaf=3, bootstrap=True, random_state=50)
classifier.fit(X_train, y_train)

# Predicting the result
y_pred = classifier.predict(X_test)

# Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
acc = accuracy_score(y_test, y_pred)
acc*=100
print("Accuracy After RandomizedSearchCV: {} %".format(acc))
print(classification_report(y_test, y_pred))

