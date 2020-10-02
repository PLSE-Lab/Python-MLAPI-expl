# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)ve

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_train.dtypes
data_train.head(5)
data_train.tail(10)

data_train['Survived'].value_counts()

data_train.groupby(['Sex','Pclass','Survived']).size()

import seaborn as sns
import matplotlib.pyplot as plt

#drawing a useless scatterplot of a categorical and an a numerical variable
ax = sns.scatterplot(data_train['Age'],data_train['Survived'])

#drawing a more useful histogram with variable variable agregatted by diferent variables
sns.set(style="darkgrid")
g = sns.FacetGrid(data_train, row="Survived", col="Sex", margin_titles=True)
bins = np.linspace(0, 80, 8)
g.map(plt.hist, "Age", color="steelblue", bins=bins)

##Creating an array to run the clasification model
train=data_train.filter(['PassengerId','Pclass','Sex','Age', 'Survived'])
X = train.iloc[:,[1, 2, 3]].values
Y = train.iloc[:,[4]].values

#to erase NaN from an array column 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(add_indicator=False, copy=True, fill_value=None,
        missing_values=np.nan, strategy='mean', verbose=0)
imputer = imputer.fit(X [:,[2]])
X [:, [2]] = imputer.transform(X [:, [2]])

#to change a categorical value in dummy variable from an array column
# applyed to the vaiable 'Sex'
# 1=male
# 0=female
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto', drop='first')
encoder.fit(X [:,[1]])
X [:,[1]] = encoder.transform(X [:,[1]]).toarray()

# Splitting the training dataset into the (Training) set and (Test) set for model evaluation porposes 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)
#To use the GaussianNB function the variable y should be shape (n,1) instead (1,n)
y_train=y_train.ravel()
y_test=y_test.ravel()

#using a Gaussian Naive Bayes to predict classes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit (X_train,y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Import data test
data_test=pd.read_csv('/kaggle/input/titanic/test.csv')
test=data_test.filter(['PassengerId','Pclass','Sex','Age'])

X_t = test.iloc[:,[1, 2, 3]].values
#correcting NaN
X_t [:, [2]] = imputer.transform(X_t [:, [2]])
#coding variable sex
X_t [:,[1]] = encoder.transform(X_t [:,[1]]).toarray()

# Predicting the Test set results
y_t = clf.predict(X_t).astype(int)

# Creating the Dataset in the submission file format
ID=test.iloc[:,0].values
output=np.column_stack((ID, y_t))


df = pd.DataFrame(data={"PassengerId": ID, "Survived": y_t})
df.to_csv("file.csv", sep=',',index=False)

#np.savetxt("file.csv", output, delimiter=",", fmt='%s')
#pd.DataFrame(output).to_csv("C:/Users/file.csv", header=header, index=None)

print('Saved file: file.csv')



