# %% [code]
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

# %% [code]
Train = pd.read_csv('/kaggle/input/texts-classification-ml-hse-2019/train.csv')
Test = pd.read_csv('/kaggle/input/texts-classification-ml-hse-2019/test.csv')
import gc

# %% [code]

# %% [code]
Train.fillna('', inplace=True)
Test.fillna('', inplace=True)

# %% [code]
Train['title&description'] = Train['title'].str[:] + ' ' + Train['description'].str[:]
Test['title&description'] = Test['title'].str[:] + ' ' + Test['description'].str[:]

# %% [code]
from sklearn.model_selection import train_test_split

X_train, y_train = Train[['title&description']], Train['Category']

del Train
gc.collect()

# %% [code]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

tf_idf = TfidfVectorizer()
tf_idf.fit(X_train['title&description'])

X_train_tf_idf = tf_idf.transform(X_train['title&description'])
test_tf_idf = tf_idf.transform(Test['title&description'])

Answer = pd.DataFrame(columns=['Id', 'Category'])
Answer['Id'] = Test['itemid']

del X_train
del Test
gc.collect()

# %% [code]
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#from lightgbm import LGBMClassifier
# %% [code]
#lr = LogisticRegression(verbose=True, n_jobs=-1, multi_class='multinomial', solver='saga')
#lr.fit(X_train_tf_idf, y_train)
gc.collect()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_jobs=-1)

knn.fit(X_train_tf_idf, y_train)
#print(accuracy_score(y_test, knn.predict(Test_tf_idf)))

# %% [code]
gc.collect()

Answer['Category'] = knn.predict(test_tf_idf)
Answer.to_csv('my_submission.csv', index=None)

"""
# %% [code]
import gc

tf_idf = TfidfVectorizer()
tf_idf.fit(Train['title&description'])
Train_tf_idf = tf_idf.transform(Train['title&description'])
Test_tf_idf = tf_idf.transform(Test['title&description'])



y = Train['Category']

Answer = pd.DataFrame(columns=['Id', 'Category'])
Answer['Id'] = Test['itemid']

del Train
del Test
gc.collect()

gb = GradientBoostingClassifier()

#lr = LogisticRegression(verbose=True, n_jobs=-1, multi_class='multinomial', solver='saga')
#lr.fit(Train_tf_idf, y)

gb.fit(Train_tf_idf, y)

Answer['Category'] = gb.predict(Test_tf_idf)
Answer.to_csv('my_submission.csv', index=None)"""