# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

from collections import OrderedDict

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


raw_emails_df = pd.read_csv('../input/emails.csv')
cv = CountVectorizer()
df_x = raw_emails_df['text']
df_y = raw_emails_df['spam']
cv_fit_x = cv.fit_transform(df_x)

x_train, x_test, y_train, y_test = train_test_split(cv_fit_x, df_y, test_size=0.3)

clf = MultinomialNB()
clf.fit(x_train, y_train)
result = clf.predict(x_test)
print(accuracy_score(y_test, result))