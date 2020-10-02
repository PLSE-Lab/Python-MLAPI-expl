#!/usr/bin/env python
# coding: utf-8

# ![photo](https://img.freepik.com/free-photo/night-view-neon-sign-with-text-words-have-power_78790-1119.jpg?size=626&ext=jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
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


df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')


# In[ ]:


import seaborn as sns
fig, plots = plt.subplots(2,3,figsize=(18,12))
plot1, plot2, plot3, plot4, plot5, plot6 = plots.flatten()
sns.countplot(df['obscene'], palette= 'deep', ax = plot1)
sns.countplot(df['threat'], palette= 'muted', ax = plot2)
sns.countplot(df['insult'], palette = 'pastel', ax = plot3)
sns.countplot(df['identity_hate'], palette = 'dark', ax = plot4)
sns.countplot(df['toxic'], palette= 'colorblind', ax = plot5)
sns.countplot(df['severe_toxic'], palette= 'bright', ax = plot6)


# In[ ]:



rslt_df = df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
rslt_df2 = df[(df['toxic'] == 1) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
new1 = rslt_df[['id', 'comment_text', 'toxic']].iloc[:23891].copy() 
new2 = rslt_df2[['id', 'comment_text', 'toxic']].iloc[:946].copy()
new = pd.concat([new1, new2], ignore_index=True)


# ### What does tf-idf mean?
# Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.

# ### How to Compute:
# 
# Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# **TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
# 
# *TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*
# 
# **IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:
# 
# *IDF(t) = log_e(Total number of documents / Number of documents with term t in it).*
# 
# See below for a simple example.
# 
# **Example:**
# 
# Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
Xv = vectorizer.fit(new['comment_text'])
import pickle


# In[ ]:


#test train split
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new["comment_text"], new['toxic'], test_size=0.33)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
X1 = vectorizer.transform(X_train)
X_test1= vectorizer.transform(X_test)


# ### SMOTE
# SMOTE (synthetic minority oversampling technique) is one of the most commonly used oversampling methods to solve the imbalance problem. It aims to balance class distribution by randomly increasing minority class examples by replicating them. SMOTE synthesises new minority instances between existing minority instances. It generates the virtual training records by linear interpolation for the minority class. These synthetic training records are generated by randomly selecting one or more of the k-nearest neighbors for each example in the minority class. After the oversampling process, the data is reconstructed and several classification models can be applied for the processed data.

# ![photo](https://miro.medium.com/max/2246/1*o_KfyMzF7LITK2DlYm_wHw.png)

# In[ ]:


print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=12)
x_train_res, y_train_res = sm.fit_sample(X1, y_train)
print('Resampled dataset shape %s' % Counter(y_train_res))


# LOGISTIC REGRESSION

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf2, x_train_res,y_train_res, cv=5,scoring='f1_weighted')


# In[ ]:


scores


# In[ ]:


y_p1 = clf2.fit(x_train_res, y_train_res).predict(X_test1)


# In[ ]:


from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p1)
print('Accuracy: %f' % accuracy)


# In[ ]:


import numpy as np

z=1.96
interval = z * np.sqrt( (0.908137 * (1 - 0.908137)) / y_test.shape[0])
interval


# > Confidence Interval - [88.97  90.21]

# SVC

# In[ ]:


from sklearn.svm import SVC
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf,x_train_res,y_train_res, cv=5)
scores


# In[ ]:


from sklearn.svm import SVC

y_p2 = clf.fit(x_train_res, y_train_res).predict(X_test1)


# In[ ]:


from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p2)
print('Accuracy: %f' % accuracy)


# In[ ]:


import numpy as np

z=1.96
interval = z * np.sqrt( (0.963279 * (1 - 0.963279)) / y_test.shape[0])
interval


# > Confidence Interval - [93.41  94.21]

# RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

clf3 = RandomForestClassifier() #Initialize with whatever parameters you want to

# 10-Fold Cross validation
scores = cross_val_score(clf3,x_train_res,y_train_res, cv=5)


# In[ ]:


scores


# In[ ]:


y_p3 = clf3.fit(x_train_res, y_train_res).predict(X_test1)


# In[ ]:


accuracy = accuracy_score(y_test, y_p3)
print('Accuracy: %f' % accuracy)


# In[ ]:


import numpy as np

z=1.96
interval = z * np.sqrt( (0.9629 * (1 - 0.9629)) / y_test.shape[0])
interval


# > Confidence Interval -  [95.94 96.74] 

# MULTINOMIAL NB

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf4 = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
scores = cross_val_score(clf4,x_train_res,y_train_res, cv=5)
y_pred4 = clf4.fit(x_train_res, y_train_res).predict(X_test1)


# In[ ]:


scores


# In[ ]:


accuracy = accuracy_score(y_test, y_pred4)
print('Accuracy: %f' % accuracy)


# In[ ]:


import numpy as np

z=1.96
interval = z * np.sqrt( (0.893376 * (1 - 0.893376)) / y_test.shape[0])
interval


# > Confidence Interval [89.06  90.38]
