# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

df = pd.read_csv('../input/spam.csv',encoding = 'latin-1')
df.head()
# Any results you write to the current directory are saved as output.

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"label", "v2":"text"})
df['label_num'] = df.label.map({'ham' : 0, 'spam' : 1})
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['length'] = df['text'].apply(lambda x: len(x))
df.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df['text'],df["label"], test_size = 0.2, random_state = 16)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words = 'english')
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)
prediction = dict()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df,y_train)
prediction["Multinomial"] = model.predict(X_test_df)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(classification_report(y_test, prediction['Multinomial'], target_names = ["Ham", "Spam"]))
accuracy_score(y_test,prediction["Multinomial"])