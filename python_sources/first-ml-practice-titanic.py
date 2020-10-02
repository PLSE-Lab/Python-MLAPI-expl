# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm,tree,linear_model, neighbors, naive_bayes,ensemble

#common model flamers
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from subprocess import check_output
#data_visualiations
import matplotlib as mpl
import matplotlib.pyplot as plot
import matplotlib.pylab as pylab
import seaborn as sb
from pandas.tools.plotting import scatter_matrix


mpl.style.use('ggplot')
sb.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print ("success")

#import train dataset to create our model

data_raw = pd.read_csv("../input/train.csv")
#import test_dataset to validate our model
data_val= pd.read_csv("../input/test.csv")
#copy train dataset so we can play with it
data_train= data_raw.copy(deep = True)

#however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data_train, data_val]
print (data_raw.info())
print (data_raw.head())
print (data_raw.sample(5))
# Any results you write to the current directory are saved as output.

#4C - Correcting, Completing, Creating, Converting
#why we need to complete null values?

print("number of null values in dataset train:\n",data_train.isnull().sum())
print ("number of null values in validation dataset:\n", data_val.isnull().sum())

#completing
#data need to be completed = column Age and Cabin. 

for dataset in data_cleaner:
    #fill na values 
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset["Embarked"].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    

#after cleaning, check data again

#print("number of null values in dataset train:\n",data_train.isnull().sum())
#print ("number of null values in validation dataset:\n", data_val.isnull().sum())
#data_train.info()

#no null values, but still we have string type data which algorithm cannot handle
#We can see that 'Name', 'Sex', 'Ticket', and 'Embarked' are all objects. In this case, they indeed are all strings. We will use Pandas built in getDummies() funciton to convert those to numbers or just drop the colunm.


#covert column 'Sex' to new Column 'Male'

data_train['Male']=pd.get_dummies(data_train['Sex'], drop_first=True)
#convert column embarked
embarked = pd.get_dummies(data_train['Embarked'], drop_first=True)
data_train = pd.concat([data_train, embarked], axis=1)
    
#drop all unneeded column
drop_column = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked','Cabin']
data_train.drop(drop_column, axis=1, inplace=True)

#4C of data is done. now, final check,

print("number of null values in dataset :\n",data_train.isnull().sum())
data_train.info()

#print("number of null values in dataset train:\n",data_train.isnull().sum())
#print ("number of null values in validation dataset:\n", data_val.isnull().sum())

#after 4C is done, it's time to build model and train
#Seperate the feature columns from the target column
X = data_train.drop('Survived', axis=1)
y = data_train['Survived']

#create the model using LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X, y)
#this will tell the model to predict the variable in dataset(feature)) to match with the target column (Survived)

#now test the prediction in dataset test, makesure you clean the data

data_val['Male']=pd.get_dummies(data_val['Sex'], drop_first=True)
#convert column embarked
embarked = pd.get_dummies(data_val['Embarked'], drop_first=True)
data_val = pd.concat([data_val, embarked], axis=1)
    
#drop all unneeded column
pas_id=data_val['PassengerId']
drop_column = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked','Cabin']
data_val.drop(drop_column, axis=1, inplace=True)

#4C of data is done. now, final check,

print("number of null values in dataset :\n",data_val.isnull().sum())
data_val.info()

#make the prediction
prediction = logmodel.predict(data_val)

#store it in CSV with PassengerId column

submission = pd.DataFrame({
        "PassengerId": pas_id,
        "Survived": prediction
    })
submission.to_csv('titanic.csv', index=False)
