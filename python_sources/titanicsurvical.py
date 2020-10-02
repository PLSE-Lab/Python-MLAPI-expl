# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation;
from sklearn.linear_model import LogisticRegression;
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer

import xgboost as xgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

plt.style.use('ggplot')

#First let's try to visualize the data

data = pd.read_csv("../input/train.csv");
#print(data)

#print column names ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
print(list(data))

#data1 = data[['SibSp','Parch']];
#print(data1);

#data.plot.bar(x='SibSp',y='Parch')



print(data.count())

cabin = pd.DataFrame()

#let's drop cabin number,Name and Ticket column
data = data.drop(['Name','Ticket'],axis=1)

#fiels embarked has many missing values so fill them up with 'NA' and then drop rows with 'NA' so number of values for each column are same i.e 714
data['Embarked'] = data['Embarked'].fillna('NA');
data['Cabin'] = data['Cabin'].fillna('U');
data = data.dropna()
print(list(data))



X = data.iloc[:,2:10]
Y = data.iloc[:,1];

#print(list(X))
print(X.count())
print(Y.count())

le = preprocessing.LabelEncoder();
X.Sex = le.fit_transform(X.Sex);
X.Embarked = le.fit_transform(X.Embarked);

pclass = pd.get_dummies( X.Pclass , prefix='Pclass' )
X = X.drop(['Pclass'],axis=1);

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = X[ 'Parch' ] + X[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

#X = X.drop(['Parch','SibSp'],axis=1);

cabin = pd.DataFrame()
X[ 'Cabin' ] = X[ 'Cabin' ].map( lambda c : c[0] );
cabin = pd.get_dummies( X['Cabin'] , prefix = 'Cabin' );
X = pd.concat( [ X , pclass,family,cabin] , axis=1 );
X = X.drop(['Cabin'],axis=1);

print(list(X))

#let's split the data into training and test data
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2);

print(X_train.count())
print(X_test.count())
print(Y_train.count())
print(Y_test.count())

#for i, C in enumerate(( 1,10,20,30,40,50,60,70,80,90,100)):

    #print(C)
lr = LogisticRegression(C=100, penalty='l2', tol=0.01,max_iter=1500);

lr.fit(X_train,Y_train)

X_pred = lr.predict(X_test)

#print(list(X_pred))
#print(list(Y_test))
print('success percentage with lr and regularization = ',100)
print(np.sum(X_pred == Y_test) / float(X_pred.size))

clf = svm.LinearSVC(C=1,penalty='l1',dual=False)
clf.fit(X_train,Y_train);
pred = clf.predict(X_test)
print('success percentage with linear svc and regularization = ', 1)
print(np.sum(pred == Y_test) / float(pred.size))

#print((X_pred == Y_test) % 100)

clf1 = RandomForestClassifier(n_estimators=50)
clf1 = clf1.fit(X_train,Y_train)
pred1 = clf1.predict(X_test)
print('success percentage with RandomForestClassifier and estimators = 50')
print(np.sum(pred1 == Y_test) / float(pred1.size))

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,Y_train) 
pred2 = neigh.predict(X_test)
print('success percentage with KNeighborsClassifier and n_neighbors = 2')
print(np.sum(pred2 == Y_test) / float(pred2.size))


testdata = pd.read_csv("../input/test.csv");
#print(testdata)
psId = np.array(testdata.PassengerId);
testdata = testdata.drop(['PassengerId','Name','Ticket'],axis=1)
print('after dropping cabin',testdata.count())


testdata['Embarked'] = testdata['Embarked'].fillna('NA');

le = preprocessing.LabelEncoder();
testdata.Sex = le.fit_transform(testdata.Sex);
testdata.Embarked = le.fit_transform(testdata.Embarked);

pclass = pd.get_dummies( testdata.Pclass , prefix='Pclass' )
testdata = testdata.drop(['Pclass'],axis=1);

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = testdata[ 'Parch' ] + testdata[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

cabin = pd.DataFrame()
testdata[ 'Cabin' ] = testdata.Cabin.fillna( 'U' )

#print(testdata['Cabin']);
testdata[ 'Cabin' ] = testdata[ 'Cabin' ].map( lambda c : c[0] );
cabin = pd.get_dummies( testdata['Cabin'] , prefix = 'Cabin' );
testdata = testdata.drop(['Cabin'],axis=1);
testdata = pd.concat( [ testdata , pclass,family,cabin] , axis=1 );


zeroArray = np.zeros(testdata.shape[0]);

cabinT = pd.DataFrame(data=zeroArray, columns=['Cabin_T'])

print(psId.shape)
print(testdata.shape)
testdata = pd.concat([testdata,cabinT],axis=1);
print(testdata.shape)
#testdata = testdata.dropna()
#print(list(X_train))
print(testdata.count)
#pred3 = neigh.predict(testdata)
pred3 = clf.predict(testdata)

finalArray = np.c_[psId,pred3];
print(finalArray.shape)
prediction = pd.DataFrame(finalArray, columns=['PassengerId','Survived']).to_csv('prediction.csv',index = False)
