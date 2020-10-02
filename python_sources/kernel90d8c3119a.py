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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

tam_veri = pd.concat([train,test])
 
tam_veri.drop('Cabin', axis = 1, inplace = True)

tam_veri.drop('Ticket', axis = 1, inplace = True)


tam_veri.drop('Name', axis = 1, inplace = True)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

age = tam_veri.iloc[:,0:1].values
age = imputer.fit_transform(age)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 

sex =  tam_veri.iloc[:, 5:6]
sex = le.fit_transform(sex)
embarked = tam_veri.iloc[:,1:2]
embarked = le.fit_transform(embarked.astype(str))

sex = pd.DataFrame(data = sex, index = range(1309), columns = ['sex'])
age = pd.DataFrame(data = age, index = range(1309), columns = ['age'])
embarked = pd.DataFrame(data = embarked, index = range(1309), columns = ['embarked'])

tam_veri.drop('Age', axis = 1, inplace = True)

tam_veri.drop('Embarked', axis = 1, inplace = True)
tam_veri.drop('Sex', axis = 1, inplace = True)

survived = tam_veri.iloc[:,5:6].values
tam_veri.drop('Survived', axis = 1, inplace = True)

sonuc = pd.concat([age, sex], axis = 1)
sonuc1 = pd.concat([sonuc, embarked], axis = 1)
tam_veri.reset_index(drop = True, inplace = True)
veri = pd.concat([tam_veri, sonuc1], axis = 1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(veri,survived, test_size = 0.319, shuffle = False)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_test = pd.DataFrame(data = y_test, index = range(418), columns = ['y_test']).values
y_train = pd.DataFrame(data = y_train, index = range(891), columns = ['y_train']).values
x_test = x_test.fillna(x_test.mean())
y_pred = lin_reg.predict(x_test)

passanger = x_test.iloc[:,2:3].values

passanger = pd.DataFrame(data = passanger, index = range(418), columns = ['PassegerID'])



from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(x_train, y_train)

y_pred2 = dtc.predict(x_test)
survived = pd.DataFrame(data = y_pred2, index = range(418), columns = ['Survived'])
son_sonuc = pd.concat([passanger, survived], axis = 1)
son_sonuc.to_csv('son_sonuc.csv')