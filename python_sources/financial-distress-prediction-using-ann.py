# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

dataset = pd.read_csv('../input/Financial Distress.csv')

# Split dataset into dependent and independent variables 

X = dataset.iloc[:,:]
y = pd.DataFrame(X.iloc[:,2])
X.drop(columns = 'Financial Distress', inplace=True)

# Encoding the categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
labelencoder_X_3 = LabelEncoder()

Companies = np.unique(X["Company"])
C = pd.DataFrame(labelencoder_X_1.fit_transform(X["Company"]))
onehotencoder = OneHotEncoder(categorical_features = [0])
C = pd.DataFrame(onehotencoder.fit_transform(C).toarray())
C.columns = Companies


Time = np.unique(X["Time"])
T = pd.DataFrame(labelencoder_X_2.fit_transform(X["Time"]))
onehotencoder = OneHotEncoder(categorical_features = [0])
T = pd.DataFrame(onehotencoder.fit_transform(T).toarray())
T.columns = Time

Features = np.unique(X["x80"])
F = pd.DataFrame(labelencoder_X_3.fit_transform(X["x80"]))
onehotencoder = OneHotEncoder(categorical_features = [0])
F = pd.DataFrame(onehotencoder.fit_transform(F).toarray())
F.columns = Features

P = pd.DataFrame(pd.concat((C,T),axis = 1))

X = X.drop(["Company","Time"],axis=1)
X = pd.DataFrame(pd.concat((P,X,F),axis = 1))

# Splitting the dataset into the Training set and Test set

y = (y<-0.5)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0,stratify = y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the keras libraries and packages

import keras 
from keras.models import Sequential 
from keras.layers import Dense

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):  
     val_predict = self.model.predict_classes(self.validation_data[0])
     val_targ = self.validation_data[1]
     _val_f1 = f1_score(val_targ, val_predict)
     _val_recall = recall_score(val_targ, val_predict)
     _val_precision = precision_score(val_targ, val_predict)
     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)
     print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
     return
 
metrics = Metrics()

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu', input_shape=(556,)))

# Adding the second hidden layer

classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu'))

# Adding the third hidden layer

classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu'))

# Adding the fourth hidden layer

classifier.add(Dense(units = 30, kernel_initializer = "uniform", activation = 'relu'))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = 'sigmoid'))

# Compiling the ANN

classifier.compile(loss='binary_crossentropy',
          optimizer= "adam",
          metrics = ['accuracy']
          )

# Fitting the ANN to the Training set

#classifier.fit(x=X_train,y=y_train,batch_size = 10, epochs = 100)

from sklearn.utils import class_weight
class_weight_real = class_weight.compute_class_weight('balanced'
                                               ,np.unique(y_train['Financial Distress'])
                                               ,y_train['Financial Distress'])

classifier.fit(X_train, y_train, 
 validation_data=(X_test, y_test),
 epochs=200,
 batch_size=200,
 callbacks=[metrics],
 class_weight=class_weight_real
 )


y_pred = classifier.predict_classes(X_test)
y_pred2 = classifier.predict_classes(X_train)


cm = confusion_matrix(y_test , y_pred)

cm2 = confusion_matrix(y_train, y_pred2)