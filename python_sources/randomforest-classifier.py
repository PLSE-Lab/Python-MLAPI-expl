# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk #scikit-learn
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

animals = pd.read_csv(os.path.join(dirname, filenames[1]))
class_ = pd.read_csv(os.path.join(dirname, filenames[0]))

#Separating Features and output and converting Dataframe to Array 
X_data = animals.iloc[:, 1:17].values # first 16 columns of data frame with all rows
y_class = animals.iloc[:,-1].values #last column of data frame with all rows

#Train and test data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_data, y_class, random_state = 0) #test_size = 0.25 random_state = 21

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

class_type = dict(zip(range(1,8),class_['Class_Type']))

y_test = np.vectorize(class_type.get)(y_test)
y_pred = np.vectorize(class_type.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Class'], colnames=['Predicted Class']))


