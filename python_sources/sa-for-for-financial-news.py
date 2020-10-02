#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


allData = pd.read_csv("/kaggle/input/sentiment-analysis-for-financial-news/all-data.csv",encoding='latin-1')


# In[ ]:


#print(allData.head())
print("Size of the dataset:" , len(allData))

allData.columns = ['sentiment', 'sentence']

print(allData.head())


# In[ ]:


import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = []

for i in range(0, len(allData)):
  sentence = re.sub('[^a-zA-Z]', ' ', str(allData['sentence'][i]))
  sentence = sentence.lower()
  sentence = sentence.split()
  ps = PorterStemmer()
  sentence = [ps.stem(word) for word in sentence if not word in set(stopwords.words('english'))]
  sentence = ' '.join(sentence)
  corpus.append(sentence)
print("size of the corpus ", len(corpus))
print("corpus: " , corpus[0])


# In[ ]:


#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1000)
X = cv.fit_transform(corpus).toarray()
y = allData.iloc[:,0].values
#print(X)
#print(X[12])
print("Dimension of Matrix :" , X.shape)
print(y)
print("Size of the result list y",len(y))


# In[ ]:


get_ipython().run_cell_magic('time', '', '#using Logistic regression with simple validation set..\nfrom sklearn.model_selection import train_test_split as tts\nx_train, x_test, y_train, y_test = tts(X, y, test_size=0.20, random_state = 0)\n\nprint("Creating a testing data set and training dataset using train_test_split lib")\n\n\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.linear_model import LogisticRegression\nprint("Importing the logistic regression and accuracy_score")\n\n\n# train a logistic regression model on the training set\n# instantiate model\nlrm_model = LogisticRegression(solver = \'newton-cg\', multi_class = \'multinomial\')\n\nprint(lrm_model)\n\n# fit model\nlrm_model.fit(x_train, y_train)\n\n# make class predictions for the testing set\nlrm_predictions = lrm_model.predict(x_test)\n\n# calculate accuracy\nmodel_accuracy = accuracy_score(y_test, lrm_predictions)\n\n# calculate accuracy on training data\nlrm_predictions_on_training_data = lrm_model.predict(x_train)\nmodel_accuracy_on_training_data = accuracy_score(y_train, lrm_predictions_on_training_data)\n\n\n#print(lrm_predictions)\n\nprint("-----------------------------------------------")\nprint("Using Logistic Regression:")\nprint("-----------------------------------------------")\nprint("Size of the Feature Vector  :", len(X[0]))\nprint()\nprint("Size of the Dataset         :",len(X))\nprint()\nprint("Accuracy score of the model :",model_accuracy)\nprint()\nprint("Accuracy score on training data:",model_accuracy_on_training_data)\nprint()\n\n\n#accuracy using confusion matrix\nfrom sklearn.metrics import confusion_matrix\nfrom matplotlib import pyplot as plt\nimport seaborn as sns\ncm = confusion_matrix(y_test,lrm_predictions)\n#print(cm)\ntmp = 0\ntt = 0\nfor i in range(0 , len(cm)):\n  for j in range(0, len(cm)):\n    tt += cm[i][j]\n    if i==j :\n      tmp += cm[i][j]\n      #print(cm[i][j])\n#print(tmp)\n#print(tt)\nprint("Accuracy score using confusion matrix: ",tmp/tt)\n\n#HeatMap\nfig, ax = plt.subplots(figsize=(15,15))\nsns.heatmap(cm, annot=True, fmt=\'d\')')


# In[ ]:


#Using Kfold for logistic regression Validation

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

lrm_model = LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial')

kf = KFold(n_splits=10, random_state=None, shuffle=False)

X = np.array(X)
y = np.array(y)
count=0
for train_index, test_index in kf.split(X):
  print(count)
  #print("TRAIN:", train_index, "TEST:", test_index)
  x_train, x_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  # fit model
  # train a logistic regression model on the training set
  # fit model
  lrm_model.fit(x_train, y_train)

  # make class predictions for the testing set
  lrm_predictions = lrm_model.predict(x_test)

  # calculate accuracy
  model_accuracy = accuracy_score(y_test, lrm_predictions)

  # calculate accuracy on training data
  lrm_predictions_on_training_data = lrm_model.predict(x_train)
  model_accuracy_on_training_data = accuracy_score(y_train, lrm_predictions_on_training_data)
  #print(lrm_predictions)
  print("-----------------------------------------------")
  print("Using Logistic Regression:")
  print("-----------------------------------------------")
  print("Size of the Feature Vector  :", len(X[0]))
  print()
  print("Size of the Dataset         :",len(X))
  print()
  print("Accuracy score of the model :",model_accuracy)
  print()
  print("Accuracy score on training data:",model_accuracy_on_training_data)
  print()

  #accuracy using confusion matrix
  cm = confusion_matrix(y_test,lrm_predictions)
  #print(cm)
  tmp = 0
  tt = 0
  for i in range(0 , len(cm)):
    for j in range(0, len(cm)):
      tt += cm[i][j]
      if i==j :
        tmp += cm[i][j]
        #print(cm[i][j])
  #print(tmp)
  #print(tt)
  #print("Accuracy score using confusion matrix: ",tmp/tt)
  count+=1

