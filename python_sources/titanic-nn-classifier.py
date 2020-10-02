import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, Dropout

#LOAD DATA
t = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

#PREPROCESS
t = t.drop(['Name', 'Ticket', 'Cabin'], axis = 1) 
t = pd.get_dummies(t, columns=['Sex'])
t = pd.get_dummies(t, columns=['Embarked'])

t = t.fillna(0)

test = test.drop(['Name', 'Ticket', 'Cabin'], axis = 1) 
test = pd.get_dummies(test, columns=['Sex'])
test = pd.get_dummies(test, columns=['Embarked'])

test = test.fillna(0)

#RESCALE FARE
f1 = t['Fare'].tolist()
f2 = test['Fare'].tolist()
f = f1 + f2
f = np.array(f)
f = f.reshape(-1, 1)

scaler = MinMaxScaler()
f = scaler.fit_transform(f)

f1s = f[0:891].tolist()
f1s = [val for sublist in f1s for val in sublist]

f2s = f[891:1309].tolist()
f2s = [val for sublist in f2s for val in sublist]

t['Fare'] = f1s
test['Fare'] = f2s

#TRAIN THE MODEL
list(t.columns)
X = t[['Pclass',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'Sex_female',
 'Sex_male',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S']]

y = t[['Survived']]

X1 = X.to_numpy()
y1 = y.to_numpy()

X_test = test[['Pclass',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'Sex_female',
 'Sex_male',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S']]

X2 = X_test.to_numpy()

model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(X1, y1, epochs = 100, batch_size = 32, verbose = 0)

y_predict = model.predict(X2)
y_predict = np.where(y_predict > 0.5, 1, 0)

sub = pd.read_csv('../input/titanic/gender_submission.csv')
sub = sub[['PassengerId']]
sub['Survived'] = y_predict
sub.to_csv('submission.csv' ,index=False)