# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn import cross_validation, preprocessing, neighbors
# Any results you write to the current directory are saved as output.

# Hi guys,
# in here i am trying to predict the intent of a gun death by taking into consideration the features as inputs.
# i am going to convert all texts into unique numbers.

df = pd.read_csv('../input/guns.csv')

df_predict = df.fillna(-99999)


def convert_text(df):
    columns = list(df)
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            print(text_digit_vals)
            df[column] = list(map(convert_to_int, df[column]))
    return df

df_predict = convert_text(df_predict)
# print(df_predict.head())


df_predict['intent'].replace(-9999999,4, inplace=True)
print(df_predict.head())

y = np.array(df_predict['intent'])
df_predict.drop('intent', 1, inplace=True)
X= np.array((df_predict).astype(float))
X= preprocessing.scale(X)

X_train, X_test, y_train, y_test= cross_validation.train_test_split(X,y, test_size=0.05)

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)











































