import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Read the data
dataset = pd.read_csv('../input/bank-additional-full.csv' , sep = ';',na_values = 'unknown')

from pandas.plotting import scatter_matrix
#scatter_matrix(dataset)
#sns.pairplot(dataset)

#plt.scatter(dataset['loan'],dataset['age'])
#plt.show()

X = dataset.iloc[:,[0,1,2,3,5,9,10,11,13]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y)

#Train data
temp = pd.DataFrame(X_train)

temp[8].isnull().sum()
temp[4].value_counts()
#Fill null values
#Here 'mode' technique is used
temp[1] = temp[1].fillna('admin.')
temp[2] = temp[2].fillna('married')
temp[3] = temp[2].fillna('university.degree')
temp[4] = temp[4].fillna('yes')
X_train = temp

#Encode the string values
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X_train[1] = lab.fit_transform(X_train[1])
X_train[2] = lab.fit_transform(X_train[2])
X_train[3] = lab.fit_transform(X_train[3])
X_train[4] = lab.fit_transform(X_train[4])
X_train[5] = lab.fit_transform(X_train[5])
y_train = lab.fit_transform(y_train)

#It is used to remove Dummy variable trap
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,2,3,4,5])
X_train = one.fit_transform(X_train)
X_train = X_train.toarray()

#Train the algorithm
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

#Test data
temp = pd.DataFrame(X_test)

temp[8].isnull().sum()
temp[4].value_counts()
#Fill null values
#Here 'mode' technique is used

temp[1] = temp[1].fillna('admin.')
temp[2] = temp[2].fillna('married')
temp[3] = temp[2].fillna('university.degree')
temp[4] = temp[4].fillna('yes')
X_test = temp

#Encoding
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X_test[1] = lab.fit_transform(X_test[1])
X_test[2] = lab.fit_transform(X_test[2])
X_test[3] = lab.fit_transform(X_test[3])
X_test[4] = lab.fit_transform(X_test[4])
X_test[5] = lab.fit_transform(X_test[5])
y_test = lab.fit_transform(y_test)

#Remvoe Dummy variable trap
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,2,3,4,5])
X_test = one.fit_transform(X_test)
X_test = X_test.toarray()

#Score
log_reg.score(X_test,y_test)