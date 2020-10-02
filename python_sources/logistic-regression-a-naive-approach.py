# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        df = pd.read_csv(os.path.join(dirname, filename))
        le = LabelEncoder()
        for column in df.columns:
            le_fit = le.fit(df[column])
            df[column] = le.fit_transform(df[column])

        X = df.drop('class', axis=1)
        Y = df['class']

        onehot = OneHotEncoder()
        X_oh = onehot.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_oh, Y, test_size=0.33, random_state=42)
        model_init = LogisticRegression()
        model = model_init.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        

# Any results you write to the current directory are saved as output.
