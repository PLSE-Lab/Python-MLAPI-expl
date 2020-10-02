# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv,sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split 

#df = pd.read_csv('../input/spam.csv', sep='\t', quoting=csv.QUOTE_NONE,names=["Status", "Message"])
df = pd.read_csv('../input/spam.csv',encoding = 'latin-1')

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"Status", "v2":"Message"})

df.loc[df["Status"]=='ham',"Status"]=1
df.loc[df["Status"]=='spam',"Status"]=0

df_x = df["Message"]
df_y = df["Status"]

cv1 = TfidfVectorizer(min_df=1,stop_words='english')

x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.2)

x_traincv = cv1.fit_transform(x_train)
a = x_traincv.toarray()

mnb = MultinomialNB()

y_train = y_train.astype('int')

mnb.fit(x_traincv,y_train)

x_testcv = cv1.transform(x_test)

pred = mnb.predict(x_testcv)

actual = np.array(y_test)

count = 0
for i in range(len(pred)):
    if pred[i] == actual[i]:
        count+=1

print ((count * 1.0)/len(pred))