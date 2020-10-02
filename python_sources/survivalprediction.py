import numpy as np

import pandas as pd

from scipy.stats.stats import pearsonr

from sklearn import preprocessing

from sklearn import tree

import subprocess

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



X_train, X_test  = train_test_split(train, test_size=0.33)

print(X_train.head(2))



# Any results you write to the current directory are saved as output.

'''

X_train = X_train.dropna(subset=['Age'])

X_test = X_test.dropna(subset=['Age'])



X_train = X_train.dropna(subset=['Age'])

X_test = X_test.dropna(subset=['Age'])

#test = test.dropna(subset=['Age']) 

X_train = X_train.dropna(subset=['Fare'])

X_test = X_test.dropna(subset=['Fare'])

#test = test.dropna(subset=['Fare'])

'''

X = X_train.iloc[:,[4,5,6,7]].copy()

Y = X_train.iloc[:, 1]



accuracy = X_test.iloc[:, [0,1]].copy()

xText = X_test.iloc[:,[4,5,6,7]].copy()



finalTest = test.iloc[:,[3,4,5,6]].copy()



print(X.head(3))

print(Y.head(3))

print(finalTest.head(3))

le = preprocessing.LabelEncoder()

le.fit(pd.unique(X.Sex))





X.Sex = le.transform(X.Sex)

xText.Sex = le.transform(xText.Sex)

finalTest.Sex = le.transform(finalTest.Sex)



X.Age.fillna(X.Age.mean(), inplace=True)

xText.Age.fillna(xText.Age.mean(), inplace=True)

finalTest.Age.fillna(finalTest.Age.mean(), inplace=True)

#X.Fare.fillna(X.Fare.mean(), inplace=True)

#xText.Fare.fillna(xText.Fare.mean(), inplace=True)

#finalTest.Fare.fillna(finalTest.Fare.mean(), inplace=True)

#plt.plot( X.Age,Y, 'bo')

#plt.ylabel('Age')

#plt.show()



print(X.head(3))

clf = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,

                               max_depth=9, min_samples_leaf=5)

clf = clf.fit( X, Y)



#print(xText.iloc[2])

#tree.export_graphviz(clf, out_file='tree.dot')

#subprocess.call(['dot', '-Tpdf', 'tree.dot', '-o' 'tree.pdf'])



isSurvived  = pd.DataFrame({'Survived' : (clf.predict(xText)),'PassengerId' : ( X_test['PassengerId'])})

finalIsSurvived = pd.DataFrame({'Survived' : (clf.predict(finalTest)),'PassengerId' : ( test['PassengerId'])})

print(finalIsSurvived.head(10))





#final.Survived = final.Survived.astype(int)

finalIsSurvived.to_csv('submit.csv', encoding='utf-8', index=False)

print(len(finalIsSurvived))



print(accuracy_score(accuracy.iloc[:,1], isSurvived.iloc[:,1]))