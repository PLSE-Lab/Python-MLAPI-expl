# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
"""
Created on Wed Jun 14 15:43:09 2017

@author: mukundhan.vijaykumar
"""

import numpy as np
import pandas as pd
import matplotlib as plt
#%matplotlib inline


#importing the data
dataset = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')


dataset.info()

#setting index 
dataset.set_index(['PassengerId'],inplace = True)

#correllating Pclass 
dataset.groupby('Pclass').Survived.mean().plot(kind = 'bar')

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
X_train = dataset[['Pclass']]
y = dataset['Survived']
X_test = test_set[['Pclass']]
classifier.fit(X_train,y)

prediction = classifier.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,index = X_test.index.values,columns=['Survived'])
contentTestPredObject1 = dfPrediction.to_csv()