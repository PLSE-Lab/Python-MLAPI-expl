# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#.......................................................................IMPORTS

from sklearn.model_selection import train_test_split #to split the dataset for training and testing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn import metrics #for checking the model accuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.svm import SVC



#..........................................................................MAIN
df = pd.read_csv("/kaggle/input/pixelss-intensity-of-positive-and-negative-nuclei/data.csv")

#Scale Features
scaler = StandardScaler()
df_data = df.loc[:, df.columns != 'Label']
scaler.fit(df_data)
scaled_features = scaler.transform(df_data.values)
scaled_features_df = pd.DataFrame(scaled_features, index=df_data.index, columns=df_data.columns)
scaled_features_df['Label'] = df['Label']

#Shuffle Dataframe
df_shuffle = scaled_features_df.iloc[np.random.permutation(len(scaled_features_df))]
df_shuffle = df_shuffle.reset_index(drop=True) #Reset the index to begin at 0

#Split Dataframe
X_train , X_test = train_test_split(df_shuffle, test_size = 0.3)
y_train = X_train['Label']
y_test = X_test['Label']
X_train.drop(['Label'], axis = 1)
X_test.drop(['Label'], axis = 1)

#Logistic Regression
clf_LR = LogisticRegression(max_iter = 1000)
clf_LR.fit(X_train, y_train)
y_pred_LR = clf_LR.predict(X_test)
accuracy_LR = metrics.accuracy_score(y_test, y_pred_LR)
print(confusion_matrix(y_test, y_pred_LR))
print('Accuracy of Logistic Regression - {:.4f}'.format(accuracy_LR))

#Support Vector Machine Classifer
clf_SVM = SVC(gamma='auto')
clf_SVM.fit(X_train, y_train)
y_pred_SVM  = clf_SVM.predict(X_test)
accuracy_SVM = metrics.accuracy_score(y_test, y_pred_SVM)
print(confusion_matrix(y_test, y_pred_SVM))
print('Accuracy of SVM - {:.4f}'.format(accuracy_SVM))

#Random Forest
clf_RF = RandomForestClassifier(n_estimators = 100, random_state=0)
clf_RF.fit(X_train, y_train)
y_pred_RF = clf_RF.predict(X_test)
accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)
print(confusion_matrix(y_test, y_pred_RF))
print('Accuracy of Random Forest - {:.4f}'.format(accuracy_RF))
