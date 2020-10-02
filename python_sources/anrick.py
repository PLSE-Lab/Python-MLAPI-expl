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

# Importing scikit
from sklearn import model_selection, preprocessing, linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings('ignore')

# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# We will use the Seaborn library
import seaborn as sns
sns.set()

import os, string
# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg'

# Loading training and test data
df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Creating the Train and Validation Dataset and preparing them for the models
X_train, X_val, y_train, y_val = train_test_split(df['text'], df['target'], test_size=0.3,
random_state=17)  

# Here we try three types of preprocessing,
# 1. Converting the text to tfidf word vectors.
# 2. Converting the text to tfidf ngram vectors.
# 3. Converting the text to tfidf character vectors.

# Converting X_train and X_val to tfidf vectors (since out models can't take text data is input)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df['text'])
xtrain_tfidf =  tfidf_vect.transform(X_train)
xvalid_tfidf =  tfidf_vect.transform(X_val)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(X_val)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(df['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_val) 

# Also creating for the X_test which is essentially test_df['text'] column
xtest_tfidf =  tfidf_vect.transform(test_df['text'])
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(test_df['text'])
xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_df['text'])

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(X_train)
xvalid_count =  count_vect.transform(X_val)
xtest_count = count_vect.transform(test_df['text'])

# We create 3 models and calculate the accuracy
model1 = linear_model.LogisticRegression()
model1.fit(xtrain_count, y_train)
accuracy=model1.score(xvalid_count, y_val)
print('Accuracy Count LR:', accuracy)
test_pred1=model1.predict(xtest_count)

model2 = linear_model.LogisticRegression()
model2.fit(xtrain_tfidf, y_train)
accuracy=model2.score(xvalid_tfidf, y_val)
print('Accuracy TFIDF LR:', accuracy)
test_pred2=model2.predict(xtest_tfidf)

model3 = linear_model.LogisticRegression()
model3.fit(xtrain_tfidf_ngram, y_train)
accuracy = model3.score(xvalid_tfidf_ngram, y_val)
print('Accuracy TFIDF NGRAM LR:', accuracy)
test_pred3 = model3.predict(xtest_tfidf_ngram)

# We ensemble the 3 models, take the most agreed upon label as the true label.
final_pred = np.array([])
for i in range(0,len(test_df['text'])):
    final_pred = np.append(final_pred, np.argmax(np.bincount([test_pred1[i], test_pred2[i], test_pred3[i]])))
    
# Creating Submission DF
sub_df = pd.DataFrame()
sub_df['Id'] = test_df['Id']
sub_df['target'] = [int(i) for i in final_pred]

# Creating submission file
sub_df.to_csv('my_submission.csv', index=False)
