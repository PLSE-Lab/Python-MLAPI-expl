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

# Read train.csv to dataframe train, test.csv to test, gender_submission.csv to y_test
train=pd.read_csv('../input/train.csv', index_col=0)
test=pd.read_csv('../input/test.csv', index_col=0)
example=pd.read_csv('../input/gender_submission.csv', index_col=0)
# Put the first column of train to y_train
y_train = train.iloc[:,0]

# Remove 'Name', 'Ticket', 'Cabin' (because i can't understand their meaning)
train1=train.iloc[:, [1,3,4,5,6,8,10]]
test1=test.iloc[:, [0,2,3,4,5,7,9]]

# Join train1 and test1 to total
total=train1.append(test1)
total.info()
# From info, 'Age' has a lot NaN, 'Sex' and 'Embarked' need to convert

# Deal with NaN in 'Age'
# I decide to use each Pclass' avg age to fillna
age1=total[total.Pclass==1].Age.mean()
age2=total[total.Pclass==2].Age.mean()
age3=total[total.Pclass==3].Age.mean()

# Use rep_age to store all the age replaced
rep_age=total[total.Pclass==1]['Age'].fillna(age1)
rep_age=rep_age.append(total[total.Pclass==2]['Age'].fillna(age2))
rep_age=rep_age.append(total[total.Pclass==3]['Age'].fillna(age3))
# Sort rep_age by index
rep_age=rep_age.sort_index()
# Replace total.age ny rep_age
total['Age']=rep_age

# fillna in 'Fare'
total['Fare']=total.Fare.fillna(total[total.Pclass==3].Fare.mean())

# Convert 'Embarked' to num
total.Embarked=total.Embarked.replace(to_replace='S', value=1)
total.Embarked=total.Embarked.replace(to_replace='Q', value=2)
total.Embarked=total.Embarked.replace(to_replace='C', value=3)
# fillna in 'Embarked' by 1, because most 'Embarked' are 1('S')
total.Embarked=total.Embarked.fillna(1)

# Convert 'Sex' to num
total.Sex=total.Sex.replace(to_replace='female', value=0)
total.Sex=total.Sex.replace(to_replace='male', value=1)
total
# Separate total to X_train and X_test
X_train=total.iloc[0:891,:]
X_test=total.iloc[891:,:]

# Start to train
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train, y_train)

# Predict X_test to y_pred
y_pred=logreg.predict(X_test)
y_pred=pd.DataFrame(y_pred, index=np.arange(892, 1310, 1), columns=['Survived'])
y_pred.index.name='PassengerId'
print(y_pred.head())
# 
#import csv
#with open('../input/gender_submission.csv', 'w', newline='') as f:
#    writer = csv.writer(f)
#    for row in y_pred:
#        writer.writerow(row)

