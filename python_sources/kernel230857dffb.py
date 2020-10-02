import pandas as pd
import numpy as np

dataset = pd.read_csv('../input/zaloni-techniche-datathon-2019/train.csv')
X = dataset.iloc[:, 0:2].values
Y = dataset.iloc[:, 2:4].values
X = X.astype(str)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y[:, 0] = labelencoder.fit_transform(Y[:, 0])
Y[:, 1] = labelencoder.fit_transform(Y[:, 1])
Y = Y.astype(float)
Y[:, 0] = Y[:, 0]*10+Y[:, 1]
Y[:, 0] = labelencoder.fit_transform(Y[:, 0])
Y = np.delete(Y, 1, axis = 1)

for i in range(0,54):
    X = np.c_[X, np.zeros(85272)]

for j in range(0,85272):
    Z = X[j,0]
    for k in range(97,123):
        count = 0
        count = Z.count(chr(k))
        X[j, 3+k-97] = count
for j in range(0,85272):
    Z = X[j,1]
    for k in range(97,123):
        count = 0
        count = Z.count(chr(k))
        X[j, 29+k-97] = count
for j in range(0,85272):
    count = 0
    Z = X[j, 0]
    count = Z.count(' ')
    X[j, 2] = count
for j in range(0,85272):
    count = 0
    Z = X[j, 1]
    count = Z.count(' ')
    X[j, 55] = count
    
X = np.delete(X, 0, axis = 1)
X = np.delete(X, 0, axis = 1)
X = X.astype(float)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 12200, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

dataset2 = pd.read_csv('../input/zaloni-techniche-datathon-2019/test.csv')
test = dataset2.iloc[:, [1,2]].values

test = test.astype(str)

for i in range(0,54):
    test = np.c_[test, np.zeros(12186)]

for j in range(0,12186):
    Z = test[j,0]
    for k in range(97,123):
        count = 0
        count = Z.count(chr(k))
        test[j, 3+k-97] = count
for j in range(0,12186):
    Z = test[j,1]
    for k in range(97,123):
        count = 0
        count = Z.count(chr(k))
        test[j, 29+k-97] = count
for j in range(0,12186):
    count = 0
    Z = test[j, 0]
    count = Z.count(' ')
    test[j, 2] = count
for j in range(0,12186):
    count = 0
    Z = test[j, 1]
    count = Z.count(' ')
    test[j, 55] = count
    
test = np.delete(test, 0, axis = 1)
test = np.delete(test, 0, axis = 1)
test = test.astype(float)

Y_pred_1 = classifier.predict(test)
Y_pred_1 = labelencoder.inverse_transform(Y_pred_1.astype(int))

Y_pred_1 = np.c_[Y_pred_1, np.zeros(12186)]
Y_pred_1[:, 1] = Y_pred_1[:,0]%10
Y_pred_1[:, 0] = (Y_pred_1[:, 0]/10).astype(int)

Y_pred_1 = Y_pred_1.astype(object)

for i in range(0, 12186):
    if( Y_pred_1[i, 0] == 0):
        Y_pred_1[i, 0] = 'f'
    else:
        Y_pred_1[i, 0] = 'm'
for i in range(0, 12186):
    if( Y_pred_1[i, 1] == 0 ):
        Y_pred_1[i, 1] = 'black'
    elif( Y_pred_1[i, 1] == 1 ):
        Y_pred_1[i, 1] = 'hispanic'
    elif( Y_pred_1[i, 1] == 2 ):
        Y_pred_1[i, 1] = 'indian'
    elif( Y_pred_1[i, 1] == 3 ):
        Y_pred_1[i, 1] = 'white'

Y_pred_1 = Y_pred_1.astype(str)
