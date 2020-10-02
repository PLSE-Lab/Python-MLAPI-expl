# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

dataset = pd.read_csv("../input/heart.csv")

X=dataset.iloc[:,:-1].values  
y = dataset.iloc[:,-1:].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y.ravel())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.03, random_state = 1,shuffle=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# -----------------------------------SVM Grid Search -------------------------------------------------#
#from sklearn import svm
#best_score = 0  
#best_params = {'C': None, 'gamma': None}
#
#C_values = [0.01, 0.03, 0.1, 0.3, 1,3, 10, 30, 100] 
#gamma_values = [0.01, 3, 10, 30, 100, 0.03, 0.1, 0.3, 1] 
#
#for C in C_values:  
#    for gamma in gamma_values:
#        print(C,gamma)
#        svc = svm.SVC(C=C, gamma=gamma,kernel = 'rbf', random_state = 1,max_iter=1000000,class_weight='balanced')
#        svc.fit(X_train, y_train)
#        score = svc.score(X_test, y_test)
#        print(score*100)
#
#        if score > best_score:
#            best_score = score
#            best_params['C'] = C
#            best_params['gamma'] = gamma
#
#print(best_score, best_params)
#-----------------------------------------------------------------------------------------------------

C=0.1
gamma=0.1

from sklearn import svm

svc = svm.SVC(C=C, gamma=gamma,kernel = 'rbf', random_state = 1,max_iter=1000000,class_weight='balanced')
svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)

print(score*100)