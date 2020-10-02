import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn import metrics
# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
#Print you can execute arbitrary python code
from sklearn.datasets import load_iris
iris=load_iris()
type(iris)
print(iris.data)

print(iris.feature_names)
print(iris.target)

print(iris.target_names)
print(type(iris.data))
print(iris.data.shape)
print(iris.target.shape)
x=iris.data
y=iris.target
knn=KNeighborsClassifier(n_neighbors=10)

print(knn)
knn.fit(x,y)
a = [[1, 2,3,4],[5,6,7,8]]
#np.asarray(a)

#a = np.array([1, 2,3,4])
print(knn.predict(a))
logreg=LogisticRegression()
logreg.fit(x,y)
y_pred=logreg.predict(x)
print(metrics.accuracy_score(y,y_pred))