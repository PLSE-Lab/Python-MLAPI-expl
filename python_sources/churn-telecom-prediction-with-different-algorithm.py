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
import numpy as np # linera Algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings

######### that code for read csv file from my directory
telecom = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')
print(telecom.head())

#Data Overview

print("Shape   ",telecom.shape)
print("Rows    ",telecom.shape[0])
print("Columns ", telecom.shape[1])
print("Features ",telecom.columns.values)
print("\n \n Missing Values  ",telecom.isnull().sum().values.sum())
print("\n uniques  values", telecom.nunique())



telecom = telecom.drop(['phone number'],axis = 1)
telecom = telecom.drop(['state'],axis = 1)

print("Shape dataset after drop some columns",telecom.shape)

print(telecom.head())

#missing value
print(telecom.isnull().sum())  #there is no missing value here


#telecom Dataset datatypes

print("telecom DATA TYPES",telecom.dtypes)
print("telecom DATA TYPES",telecom.dtypes.value_counts())

#telecom data information

print("Info of dataset ",telecom.info())

#describe our dataset

print("describe ",telecom.describe())


#data cleaning
#data replace with 0 and 1
telecom.internationalplan.replace(('yes', 'no'), (1, 0), inplace=True)

#data replace with 0 and 1
telecom.voicemailplan.replace(('yes', 'no'), (1, 0), inplace=True)
print(telecom.head())

#convert churn True False into 0 and 1
telecom.churn = telecom.churn.astype(int)
print(telecom.head())

#on the above we will check all data is clean


#change scaler for all data data convert in to float my show error some value not convert into float so i used this things

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(telecom.drop('churn',axis=1))
scaled_features = scaler.transform(telecom.drop('churn',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=telecom.columns[:-1])
print(df_feat.head())

# using KNN Classifier
x = telecom.iloc[:, :-1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, telecom['churn'],

                                                    test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print("KNN prediction",pred)

#import for confusion matrix and classifiaction report
from sklearn.metrics import classification_report,confusion_matrix

print("confusion_matrix for KNN \n",confusion_matrix(y_test,pred))
print("KNN classification_report\n ",classification_report(y_test,pred))


#Now we will used logistic regression


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

pred = logmodel.predict(X_test)
print("Logistic prediction\n",pred)

print("confusion_matrix for Logistic  \n",confusion_matrix(y_test,pred))
print("Logistic classification_report\n ",classification_report(y_test,pred))

#now we use linear regression


from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train,y_train)

pred = reg.predict(X_test)
print("Linear regression \n",pred)

plt.scatter(y_test,pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


#now we will used DecisionTree

#now we used DecisionTreeClassifier


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)
print("DecisionTreeClassifier \n",pred)

print("confusion_matrix for  DecisionTreeClassifier \n",confusion_matrix(y_test,pred))
print("DecisionTreeClassifier classification_report\n ",classification_report(y_test,pred))


#now we used Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

pred = rfc.predict(X_test)
print("Random forest prediction \n",pred)

print("confusion_matrix for  Random Forest \n",confusion_matrix(y_test,pred))
print("DecisionTreeClassifier Random Forest\n ",classification_report(y_test,pred))
