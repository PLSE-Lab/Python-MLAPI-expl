# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# import data 
data=pd.read_csv("../input/train.csv", sep=",")
print(data.info())
y=data.pop('label')

# train test split
X_train, X_test, y_train, y_test= train_test_split(data,y, test_size=0.2, random_state=42)
print(X_train.info())
print(X_test.info())

# PCA
pca=PCA(n_components=40)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)
print("Explained variance")
print(sum(pca.explained_variance_ratio_))

# Models
rf=RandomForestClassifier(n_estimators=1000, n_jobs=4)
knn=KNeighborsClassifier(weights="distance")
svm=SVC(C=0.1, kernel="poly")


# Random Forest PCA
rf.fit(X_train_pca, y_train)
print("RF")
print("Accuracy Train")
print(rf.score(X_train_pca, y_train))
print("Accuracy Test")
print(rf.score(X_test_pca, y_test))

# KNN 
knn.fit(X_train_pca, y_train)
print("KNN")
print("Accuracy Train")
print(knn.score(X_train_pca, y_train))
print("Accuracy Test")
print(knn.score(X_test_pca, y_test))

# SVM
svm.fit(X_train_pca, y_train)
print("SVM")
print("Accuracy Train")
print(svm.score(X_train_pca, y_train))
print("Accuracy Test")
print(svm.score(X_test_pca, y_test))

# knn gridsearch 
para_grid={'n_neighbors': [5,10], 
        'weights': ['uniform', 'distance']
    }

gs=GridSearchCV(estimator=knn, param_grid=para_grid, n_jobs=8, scoring='accuracy')
gs.fit(X_train_pca, y_train)
print("Gridsearch KNN")
print("Accuracy Train")
print(gs.score(X_train_pca, y_train))
print("Accuracy Test")
print(gs.score(X_test_pca, y_test))
print(gs.best_params_)

# stacking 
stack=VotingClassifier(estimators=[("rf",rf), ("knn", knn), ("svm", svm)])
stack.fit(X_train_pca, y_train)
print("Stacked models")
print("Accuracy Train")
print(stack.score(X_train_pca, y_train))
print("Accuracy Test")
print(stack.score(X_test_pca, y_test))



# predict 
predict_data=pd.read_csv("../input/test.csv", sep=",")
predict_data_pca=pca.transform(predict_data)
y_predict=stack.predict(predict_data_pca)
image_id=range(1, len(predict_data)+1)

# write to csv
submission=pd.DataFrame({'ImageId': image_id, 'Label': y_predict})
print(submission.isnull().sum())
submission.to_csv('stacked_model.csv', sep=',', index=False)