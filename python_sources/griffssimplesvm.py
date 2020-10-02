#so-so accuracy, but I wanted to try this out
import numpy as np 
import pandas as pd 

from sklearn import svm
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/diabetes.csv')

X = np.array(df.drop(['Outcome'], 1))
y = np.array(df['Outcome'])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)

accuracy = clf.score(X_test,y_test)
print('Accuracy from the split training and test: ' + str(accuracy))
print('\n')