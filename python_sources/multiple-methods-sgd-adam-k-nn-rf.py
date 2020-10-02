# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD,Adam

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Feature matrix
train_data = train.iloc[:,:561].as_matrix()
test_data = test.iloc[:,:561].as_matrix()

train_labels = train.iloc[:,562:].as_matrix()
test_labels = test.iloc[:,562:].as_matrix()


train_labelss=np.zeros((len(train_labels),6))
test_labelss=np.zeros((len(test_labels),6))



for k in range (0,len(train_labels)):
    if train_labels[k] =='STANDING':
        train_labelss[k][0]=1
    elif train_labels[k] =='WALKING':
        train_labelss[k][1]=1
    elif train_labels[k] =='WALKING_UPSTAIRS':
        train_labelss[k][2]=1
    elif train_labels[k] =='WALKING_DOWNSTAIRS':
        train_labelss[k][3]=1
    elif train_labels[k] =='SITTING':
        train_labelss[k][4]=1
    else:
        train_labelss[k][5]=1

for k in range (0,len(test_labels)):
    if test_labels[k] =='STANDING':
        test_labelss[k][0]=1
    elif test_labels[k] =='WALKING':
        test_labelss[k][1]=1
    elif test_labels[k] =='WALKING_UPSTAIRS':
        test_labelss[k][2]=1
    elif test_labels[k] =='WALKING_DOWNSTAIRS':
        test_labelss[k][3]=1
    elif test_labels[k] =='SITTING':
        test_labelss[k][4]=1
    else:
        test_labelss[k][5]=1

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=561))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(train_data, train_labelss,nb_epoch=30,batch_size=128)
score = model.evaluate(test_data, test_labelss, batch_size=128)
print(score)


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_data, train_labelss,nb_epoch=30,batch_size=128)
score = model.evaluate(test_data, test_labelss, batch_size=128)
print(score)

###### Random Forest #######
trainData  = train.drop('Activity' , axis=1).values
trainLabel = train.Activity.values

testData  = test.drop('Activity' , axis=1).values
testLabel = test.Activity.values

encoder = LabelEncoder()

# encoding test labels 
encoder.fit(testLabel)
testLabelEncoder = encoder.transform(testLabel)

# encoding train labels 
encoder.fit(trainLabel)
trainLabelEncoder = encoder.transform(trainLabel)

rf = RandomForestClassifier(n_estimators=200,  n_jobs=4, min_samples_leaf=10)    
#train
rf.fit(trainData, trainLabelEncoder)

y_te_pred = rf.predict(testData)

acc = accuracy_score(testLabelEncoder, y_te_pred)
print("Random Forest Accuracy: %.5f" % (acc))

##### K-Nearest Neighbors ######
clf = KNeighborsClassifier(n_neighbors=24)

knnModel = clf.fit(trainData , trainLabelEncoder)
y_te_pred = clf.predict(testData)

acc = accuracy_score(testLabelEncoder, y_te_pred)
print("K-Nearest Neighbors Accuracy: %.5f" % (acc))