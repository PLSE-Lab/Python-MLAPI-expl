# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Import Datasets
train_set = pd.read_csv('../input/titanic/train.csv')
test_set = pd.read_csv('../input/titanic/test.csv')

#Splitting traing set into dependent and independent variable
features = ['Sex', 'Pclass', 'SibSp', 'Parch']
x_train = train_set[list(features)].values
y_train = train_set.Survived.values[:, None].tolist()
x_test = test_set[list(features)].values

#For training set
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough' )
x_train = np.array(ct.fit_transform(x_train))

#For test set
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough' )
x_test = np.array(ct.fit_transform(x_test))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x1, x2, y1, y2 = train_test_split(x_train, y_train, test_size = 0.3, random_state = 0)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x1, y1)

# Predicting the Test set results
y_pred = classifier.predict(x2)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
cm = confusion_matrix(y2, y_pred)
acc_score = accuracy_score(y2, y_pred)
print(f'accuracy score obtained by fitting K-NN is: {round(acc_score*100, 2)}%')
cr = classification_report(y2, y_pred) 
print(cr)

#Now applying the Kernel SVM to the test set 
y_test_pred = classifier.predict(x_test) 
y_test_1 = pd.DataFrame(y_test_pred, columns = ['Survived'])

# Concatenating passenger id and predicted values
final_result = pd.concat([test_set['PassengerId'], y_test_1], axis = 1)

#Visualizing the test result and finding out the number of deaths
sns.countplot(y_test_pred)
unique, counts = np.unique(y_test_pred, return_counts=True)
dict(zip(unique, counts))