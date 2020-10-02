# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



result = pd.DataFrame()

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")

train.head(10)
s = test

s['Family'] = s['SibSp'] + s['Parch']
#s[['Embarked','Sex','Pclass','Survived']].groupby(['Embarked','Sex','Pclass']).mean().head(71)
group = s.query('(Pclass == 1 or Pclass == 2) and Sex == \'female\'')
group['Alive'] = 1
result = pd.concat([group, result])

group = s.query('Pclass == 3 and Sex == \'female\' and (Embarked == \'C\' or Embarked == \'Q\')')
group['Alive'] = 1
result = pd.concat([group, result])

group = s.query('Pclass == 3 and Sex == \'male\' and Embarked == \'C\' and Family > 0')
group['Alive'] = 1

result = pd.concat([group, result])
result = result[['PassengerId','Alive']]

ret = pd.merge(s, result, on='PassengerId', how='outer')[['PassengerId','Family','Alive']].fillna(0)

ret.columns = ['PassengerId','Family','Survived']
ret.loc[ret.Family > 3,'Survived'] = 0
print(ret.tail(10))
ret.Survived = ret.Survived.astype(np.int64)
ret.fillna(0)[['PassengerId','Survived']].to_csv('output.csv',index = False)