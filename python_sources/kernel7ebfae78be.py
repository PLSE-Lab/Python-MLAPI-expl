import numpy as np 
import pandas as pd
import tensorflow as  tf

train  = pd.read_csv("../input/titanic/train.csv")
test  = pd.read_csv("../input/titanic/test.csv")
Y_train = train["Survived"]

def process_data(data):
    X_train = data[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    X_train = X_train.fillna(X_train.mean())

    X_train.loc[:,'Sex'] = X_train['Sex'].replace(['female','male'],[0,1]).values
    X_train.loc[:,'Pclass'] = X_train.loc[:,'Pclass'] - 1

    sex_cols = tf.keras.utils.to_categorical(X_train['Sex'], num_classes=2)
    class_cols = tf.keras.utils.to_categorical(X_train['Pclass'], num_classes=3)

    return X_train

X_train = process_data(train)
X_test = process_data(test)
X_test.head()

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.utils import np_utils
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

y_train_categorical = np_utils.to_categorical(Y_train)

model = Sequential()
model.add(Dense(64,input_dim=6))
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
a = model.fit(X_train, y_train_categorical, epochs=500, batch_size = 64)
preds = model.predict_classes(X_test.values)
test_predict = np.where(preds>0,1,0)
PassengerId = test['PassengerId']
submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': test_predict.ravel() })
submission.to_csv("submission.csv", index=False)