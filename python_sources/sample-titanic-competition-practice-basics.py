# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
datatest=pd.read_csv("../input/test.csv")
label_encoder=LabelEncoder()

X = data.iloc[:,[2,4,5,6,7,9,11]].fillna('0')
Xtest=datatest.iloc[:,[1,3,4,5,6,8,10]].fillna('0')

#label encode both
X['Sex'] = label_encoder.fit_transform(X['Sex'])
X['Embarked'] = label_encoder.fit_transform(X['Embarked'])

Xtest['Sex'] = label_encoder.fit_transform(Xtest['Sex'])
Xtest['Embarked'] = label_encoder.fit_transform(Xtest['Embarked'])

#hot encode column number 6
oneHotEncoder=OneHotEncoder(categorical_features=[0,6])
X=oneHotEncoder.fit_transform(X.values).toarray()
Xtest=oneHotEncoder.fit_transform(Xtest.values).toarray()

#delete the column causing dummy variable trap
X=np.delete(X,3,1)
Y = data.iloc[:,1]

#do feature scaling 
scx=StandardScaler()
X=pd.DataFrame(scx.fit_transform(X))
Xtest=pd.DataFrame(scx.fit_transform(Xtest))
#train the multivariable linear regression algo
regressor=LogisticRegression(random_state=0)
regressor.fit(X,Y)
#predicted y values are as follows
Ypred=regressor.predict(Xtest)

data_to_submit=pd.DataFrame({'PassengerId':datatest.values[:,0],'Survived':Ypred[0:]})
data_to_submit.to_csv('csv_to_submit.csv', index = False)
