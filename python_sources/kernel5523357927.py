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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
d1 = pd.read_csv("../input/zaloni-techniche-datathon-2019/train.csv")
y = d1['gender']
y1 = d1['race']
from sklearn.feature_extraction.text import CountVectorizer as cv
d1.fillna("A",inplace=True)
name1 = d1['first_name']
name1 = name1.astype(str)
X1 = name1.apply(lambda x: x.split()[0])
X2 = d1['last_name']
X3 = X1 + ' ' +X2
v = cv()
d2 = pd.read_csv("../input/zaloni-techniche-datathon-2019/test.csv")
d2.fillna("0",inplace=True)
namex1 = d2['first_name']
namex1 = namex1.astype(str)
Xx1 = namex1.apply(lambda x: x.split()[0])
Xx2 = d2['last_name']
Xx3 = Xx1 + ' ' +Xx2
XX = np.concatenate((X3,Xx3),axis=0)
X = v.fit_transform(XX)

Xtr,Xte = train_test_split(X, test_size=12186/97458, shuffle=False)
clf = tree.DecisionTreeClassifier()
clf.fit(Xtr, y)

clf1 = LR()
clf1.fit(Xtr,y1)

gender = np.array(clf.predict(Xte))
race = np.array(clf1.predict(Xte))

id = np.array(range(1,12187))

dict = {'id':id, 'gender':gender, 'race':race}
df = pd.DataFrame(dict)
df.to_csv('../input/final.csv')
