import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Train Dataset
dataset_train = pd.read_csv('../input/kaggle-titanic-master/kaggle-titanic-master/input/train.csv')

X_train = dataset_train.iloc[:,[2,4,5]].values
y_train = dataset_train.iloc[:,1].values

y_train = y_train.reshape(-1,1)

#plt.scatter(X_train[:,1],y_train)
#plt.show()
'''
Fill missing value or use Imputer
temp = pd.DataFrame(X_train[:,:3])
temp[0].value_counts()
temp[2].isnull().sum()
#temp[1] = temp[1].fillna('male')
X_train[:,:3] = temp
dataset_train[5] = temp[2]
del(temp)'''

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN',strategy='most_frequent')
X_train[:,[2]] = imp.fit_transform(X_train[:,[2]])
y_train = imp.fit_transform(y_train)

#Encoding Gender
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X_train[:,1] = lab.fit_transform(X_train[:,1])

#Applying DecisionTree Algorithm
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 5)
dtf.fit(X_train,y_train)

'''Check score on training dataset'''
#dtf.score(X_train,y_train)

# Test Dataset
dataset_test = pd.read_csv('../input/kaggle-titanic-master/kaggle-titanic-master/input/test.csv')

X_test = dataset_test.iloc[:,[1,3,4]].values

#It is used to check whether 'X' contains NaN values or not.
'''
temp = pd.DataFrame(X_test[:,:3])
temp[0].isnull().sum()
temp[1].isnull().sum()
temp[2].isnull().sum()
del(temp)'''

#Filling missing values using most_frequent as strategy
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN',strategy='most_frequent')
X_test[:,[2]] = imp.fit_transform(X_test[:,[2]])

#Again encoding gender
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X_test[:,1] = lab.fit_transform(X_test[:,1])

#Predict values for test dataset
y_pred = dtf.predict(X_test)

y_data = pd.read_csv('../input/kaggle-titanic-master/kaggle-titanic-master/input/gender_submission.csv')
y_test = y_data.iloc[:,1].values
dtf.score(X_test,y_test)

'''Creating confusion matrix.
It is used to calculate multiple scores.'''
from sklearn.metrics import confusion_matrix
con = confusion_matrix(y_test,y_pred)

from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)