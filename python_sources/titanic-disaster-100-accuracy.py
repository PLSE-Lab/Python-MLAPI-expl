# import libraries
import pandas as pd 
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn import svm
import csv
from sklearn.preprocessing import normalize

# import Data from file CSV
x = pd.read_csv("train.csv")
y = pd.read_csv("test.csv")
z = pd.read_csv("gender_submission.csv")

# choice of data which is useful for my opinion
df = pd.DataFrame(x,columns = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"])
df1 = pd.DataFrame(y,columns = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"])
Y_train = pd.DataFrame(x,columns = ["Survived"])
Y_test = pd.DataFrame(y,columns = ["Survived"])
z = pd.DataFrame(z,columns = ["Survived"])
z1 = pd.DataFrame(y,columns = ["PassengerId"])

# convert char to the numeric values
new_df = df.replace(["male","female","S","C","Q"],["1","0","1","2","3"])
new_df1 = df1.replace(["male","female","S","C","Q"],["1","0","1","2","3"])
X_train = new_df.fillna(value = 0,axis = 0,inplace = False)
X_test = new_df1.fillna(value = 0,axis = 0,inplace = False)

# making of train and  test data
X = np.array(X_train)
X = normalize(X,axis = 1,norm='l1')
X_test = np.array(X_test)
X_test = normalize(X_test,axis = 1,norm='l1')
Y = np.array(Y_train)

Y_test = np.array(Y_test)
z = np.array(z)
z1 = np.array(z1)

X = X.astype(float)
Y = Y.astype(float)
X_test = X_test.astype(float)
Y_test = Y_test.astype(float)

# applying linear SVM and find out the accuracy
clf = svm.SVC(kernel = 'linear')
clf.fit(X,Y)
predicted = clf.predict(X_test)
report = classification_report(Y_test, predicted)
accuracy = clf.score(X_test,Y_test)
print(accuracy*100)
print(report)
preds = clf.predict(X_test)
Y_test = np.array(Y_test)

# save the file 
df = pd.DataFrame({"Survived":preds,"PassengerId":X_8})
df.to_csv('output.csv',index = False)
print(df)






