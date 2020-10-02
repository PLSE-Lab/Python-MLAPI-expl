ent # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/ts-accidents'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(dirname +'/'+ 'Accident_train.csv')

X = train_data.drop(['Collision_Ref_No','Policing_Area','Collision_Severity', 'Weekday_of_Collision'], axis=1 )
X = np.array(X.fillna(train_data.mean()))
print(X.shape)
array = train_data.values
class_lab = train_data["Collision_Severity"]
print(class_lab.shape)
print(class_lab)

#test dataset
test_data = pd.read_csv( dirname +'/' + 'Accident_test.csv')
X_test =  test_data.drop(['Collision_Ref_No','Policing_Area','Collision_Severity', 'Weekday_of_Collision'], axis=1 )
X_test = np.array(X_test.fillna(X_test.mean()))

Y_test = test_data['Collision_Severity']

Y_test = Y_test.replace(to_replace ="Predict", value = 0)


num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

model.fit(X, class_lab)
predictions = model.predict(X_test)
# print(predictions)

# print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))

df = pd.DataFrame(predictions)

df.to_csv('RandomForest_predict.csv')



