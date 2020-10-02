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
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

#Feature matrix
train_features = train.iloc[:,:561].as_matrix()
test_features = test.iloc[:,:561].as_matrix()

train_results = train.iloc[:,562:].as_matrix()
test_results = test.iloc[:,562:].as_matrix()
train_resultss=np.zeros((len(train_results),6))
test_resultss=np.zeros((len(test_results),6))
print(train_resultss)
for k in range (0,len(train_results)):
    if train_results[k] =='STANDING':
        train_resultss[k][0]=1
    elif train_results[k] =='WALKING':
        train_resultss[k][1]=1
    elif train_results[k] =='WALKING_UPSTAIRS':
        train_resultss[k][2]=1
    elif train_results[k] =='WALKING_DOWNSTAIRS':
        train_resultss[k][3]=1
    elif train_results[k] =='SITTING':
        train_resultss[k][4]=1
    else:
        train_resultss[k][5]=1

for k in range (0,len(test_results)):
    if test_results[k] =='STANDING':
        test_resultss[k][0]=1
    elif test_results[k] =='WALKING':
        test_resultss[k][1]=1
    elif test_results[k] =='WALKING_UPSTAIRS':
        test_resultss[k][2]=1
    elif test_results[k] =='WALKING_DOWNSTAIRS':
        test_resultss[k][3]=1
    elif test_results[k] =='SITTING':
        test_resultss[k][4]=1
    else:
        test_resultss[k][5]=1


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=561))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_features, train_resultss,epochs=30,batch_size=128)
score = model.evaluate(test_features, test_resultss, batch_size=128)
print(score)
