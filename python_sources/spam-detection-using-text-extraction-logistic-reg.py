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
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper

df = pd.read_csv(r'/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding = "ISO-8859-1")

#check for any null value if any thing ==1 then there it is
df.isnull().sum()

#check for unique
df['v1'].unique()
df['v1'].value_counts()

#assign  column  names
train = df[['v1', 'v2']]
train.columns = ['label','message']
#find the length of the message
train['length'] = train['message'].apply(lambda x: len(x))

#distribution of length
#plot shopw that there is a specific range of length which shows message as spam
plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(train[train['label']=='ham']['length'],bins=bins,alpha =0.8)
plt.hist(train[train['label']=='spam']['length'],bins=bins,alpha =0.8)
plt.legend(('ham','spam'))
plt.show()

#text feature extraction
mapper = DataFrameMapper([
    ('label',LabelEncoder()),
    ('message', TfidfVectorizer()),
    ('length', None)
],df_out=True)

df_train_mapper = mapper.fit_transform(train)
x= df_train_mapper.loc[:, df_train_mapper.columns != 'label']
y = df_train_mapper['label']

#split dataset into train-test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3,random_state=42)
x_train.shape

#build model and predict
from sklearn.linear_model import  LogisticRegression
model = LogisticRegression(solver='lbfgs')
model.fit(x_train,y_train)

predictions = model.predict(x_test)

#test accuracy of the model
from sklearn import  metrics
print(metrics.confusion_matrix(y_test,predictions))

#beautify confusion matrix
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print(metrics.classification_report(y_test,predictions))
