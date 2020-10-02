# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
random.seed(90)
from sklearn.neural_network import MLPClassifier
print('Random',random.random())
import matplotlib as plt
from sklearn.preprocessing import StandardScaler  
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from numpy import corrcoef, sum, log, arange
from numpy.random import rand
from pylab import pcolor, show, colorbar, xticks, yticks
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as svm
from sklearn.cross_validation import cross_val_score, train_test_split
import numpy as np
from sklearn import preprocessing,cross_validation,svm,neighbors
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/UCI_Credit_Card.csv')

df = data.copy()
target = 'default payment next month'
print(df.columns)

def describe_factor(x):
    ret = dict()
    for lvl in x.unique():
        if pd.isnull(lvl):
            ret["NaN"] = x.isnull().sum()
        else:
           ret[lvl] = np.sum(x==lvl)
    return ret

print('Sex')
print(describe_factor(df['SEX']))
# {1: 11888, 2: 18112}

print('Education is ordinnal Keep it, but set, others to NA')
print(describe_factor(df["EDUCATION"]))
# {0: 14, 1: 10585, 2: 14030, 3: 4917, 4: 123, 5: 280, 6: 51}


df["EDUCATION"] = df["EDUCATION"].map({0: np.NaN, 1:1, 2:2, 3:3, 4:np.NaN, 
    5: np.NaN, 6: np.NaN})
print(describe_factor(df["EDUCATION"]))
# {1.0: 10585, 2.0: 14030, 3.0: 4917, 'NaN': 468}

print('MARRIAGE 0,3=>NA')
print(describe_factor(df["MARRIAGE"]))
# {0: 54, 1: 13659, 2: 15964, 3: 323}

df.MARRIAGE = df.MARRIAGE.map({0:np.NaN, 1:1, 2:0, 3:np.NaN})
print(describe_factor(df.MARRIAGE))
# {0.0: 15964, 1.0: 13659, 'NaN': 377}

print("Others are quantitative and presents")

print(df.describe())


print(df.isnull().sum())


df.ix[df["EDUCATION"].isnull(), "EDUCATION"] = df["EDUCATION"].mode()
df.ix[df["MARRIAGE"].isnull(), "MARRIAGE"] = df["MARRIAGE"].mode()
print(df.isnull().sum().sum())


describe_factor(df[target])
{0: 23364, 1: 6636}


predictors = df.columns.drop(['ID', target])
X = np.asarray(df[predictors])
y = np.asarray(df[target])

data=X[:300,:].transpose()
R = corrcoef(data)
pcolor(R)
colorbar()
yticks(arange(0,21),range(0,22))
xticks(arange(0,21),range(0,22))
show()
X = X
Y =y
model = LogisticRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X, Y)
print("Num Features:",fit.n_features_)
print("Selected Features:",fit.support_)
print("Feature Ranking: ",fit.ranking_)
X=pd.DataFrame(X)


X1=X[[1,2,3,4,5,6,7,8,9,10,12,18]]

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=0, stratify=y)

X_train=preprocessing.robust_scale(X_train, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
#
X_test=preprocessing.robust_scale(X_test, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)


clf=svm.LinearSVC(C=1.0,random_state=0)
clf.fit(X_train,y_train)


accuracy=clf.score(X_test,y_test)
print("SVM using Linear SVC accuracy",accuracy)

clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print("Decision Tree accuracy",accuracy)

clf= RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=1, verbose=0, warm_start=False, class_weight=None)
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print("Random Forest",accuracy)


clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print('KNN Accuracy',accuracy)

clf=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5,2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.001, validation_fraction=0.1, verbose=False,
       warm_start=False)


clf.fit(X_train,y_train)
##
accuracy=clf.score(X_test,y_test)
print("ANN",accuracy)

