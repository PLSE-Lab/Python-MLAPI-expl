#!/usr/bin/env python
# coding: utf-8

# ![image](https://blogs.sas.com/content/subconsciousmusings/files/2018/10/flupredictionLIME.jpg)

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



rslt_df = df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
rslt_df2 = df[(df['toxic'] == 1) | (df['severe_toxic'] == 1) | (df['obscene'] == 1) | (df['threat'] == 1) | (df['insult'] == 1) | (df['identity_hate'] == 1)]
new1 = rslt_df[['id', 'comment_text', 'toxic']].iloc[:23891].copy() 
new2 = rslt_df2[['id', 'comment_text', 'toxic']].iloc[:946].copy()
new = pd.concat([new1, new2], ignore_index=True)


# In[ ]:


new.tail()


# In[ ]:


#test train split
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new["comment_text"], new['toxic'], test_size=0.33)


# ##### Tf/Idf

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
X1 = vectorizer.fit_transform(X_train)
X_test1= vectorizer.transform(X_test)


# In[ ]:


class_names = ['nontoxic', 'toxic']


# ##### Logistic Regression

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf2, X1,y_train, cv=5,scoring='f1_weighted')


# In[ ]:


y_p1 = clf2.fit(X1, y_train).predict(X_test1)


# In[ ]:


from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p1)
print('Accuracy: %f' % accuracy)


# ## LIME

# For reference please see the [paper](https://arxiv.org/abs/1602.04938) 

# In[ ]:


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, clf2)


# In[ ]:


new["comment_text"][0]


# In[ ]:


print(c.predict_proba([new["comment_text"][0]]))


# > here we have to pass the raw text and it gives the prob for non-toxic and toxic class for perticular comment at index 0

# ## Example1

# ### EXPLAINING PREDICTION USING LIME

# In[ ]:


from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# In[ ]:


X_test = X_test.tolist()


# In[ ]:


X_test[0]


# In[ ]:


type(y_test)


# In[ ]:


y_test = y_test.tolist()


# In[ ]:


y_test = np.array(y_test)


# In[ ]:


type(y_test)


# > we have converted X_test to list and y_test to ndarray because explain_instance takes input like this.

# We will now  generate an explanation with at most 10 features for an arbitrary document in the test set.

# In[ ]:


idx = 0
exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


# The classifier got this example right (it predicted non-toxic).
# The explanation is presented below as a list of weighted features.

# features contributing for classification

# In[ ]:


exp.as_list()


# In[ ]:


print('Original prediction:', clf2.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['you']] = 0
tmp[0,vectorizer.vocabulary_['thanks']] = 0
print('Prediction removing some features:', clf2.predict_proba(tmp)[0,1])
print('Difference:', clf2.predict_proba(tmp)[0,1] - clf2.predict_proba(X_test1[idx])[0,1])


# These weighted features are a linear model, which approximates the behaviour of the logistic Regression classifier in the vicinity of the test example. Roughly, if we remove 'you' and 'thanks' from the document , the prediction should move towards the opposite class by about the sum of the weights for both features.

# #### Visualizing explanations non-toxic comment on unbalanced dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi.html')


# In[ ]:


exp.show_in_notebook(text=True)


# OVERSAMPLE

# In[ ]:


print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=12)
x_train_res, y_train_res = sm.fit_sample(X1, y_train)
print('Resampled dataset shape %s' % Counter(y_train_res))


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf, x_train_res,y_train_res, cv=5,scoring='f1_weighted')


# In[ ]:


y_p2 = clf.fit(x_train_res, y_train_res).predict(X_test1)


# In[ ]:




from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p2)
print('Accuracy: %f' % accuracy)


# In[ ]:


from lime import lime_text
from sklearn.pipeline import make_pipeline
c2 = make_pipeline(vectorizer, clf)


# In[ ]:


print(c2.predict_proba([new["comment_text"][0]]))


# In[ ]:


idx = 0
exp = explainer.explain_instance(X_test[idx], c2.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c2.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


print('Original prediction:', clf.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['article']] = 0
tmp[0,vectorizer.vocabulary_['you']] = 0
print('Prediction removing some features:', clf.predict_proba(tmp)[0,1])
print('Difference:', clf.predict_proba(tmp)[0,1] - clf.predict_proba(X_test1[idx])[0,1])


# #### Visualizing explanations for non-toxic comment after Oversampling the dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi.html')


# In[ ]:


exp.show_in_notebook(text=True)


# UNDERSAMPLE

# In[ ]:


from imblearn.under_sampling import NearMiss
nm = NearMiss()
X_d, y_d = nm.fit_resample(X1, y_train)
print('Resampled dataset shape %s' % Counter(y_d))


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf1, X_d,y_d, cv=5,scoring='f1_weighted')


# In[ ]:


y_p3 = clf1.fit(X_d, y_d).predict(X_test1)


# In[ ]:


from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p3)
print('Accuracy: %f' % accuracy)


# In[ ]:


from lime import lime_text
from sklearn.pipeline import make_pipeline
c3 = make_pipeline(vectorizer, clf1)


# In[ ]:


print(c3.predict_proba([new["comment_text"][0]]))


# In[ ]:


idx = 0
exp = explainer.explain_instance(X_test[idx], c3.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c3.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


print('Original prediction:', clf1.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['article']] = 0
tmp[0,vectorizer.vocabulary_['you']] = 0
print('Prediction removing some features:', clf1.predict_proba(tmp)[0,1])
print('Difference:', clf1.predict_proba(tmp)[0,1] - clf1.predict_proba(X_test1[idx])[0,1])


# #### Visualizing explanations for non-toxic comment after Undersampling the dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi.html')


# In[ ]:


exp.show_in_notebook(text=True)


# # Example 2

# NOW for TOXIC Comment 

# In[ ]:


#printing ids of comments which are toxic
count=-1
for x in y_test:
    count=count+1
    if x==1:
        print(count)


# In[ ]:


idx = 141
exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


print('Original prediction:', clf2.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['shit']] = 0
tmp[0,vectorizer.vocabulary_['bastard']] = 0
print('Prediction removing some features:', clf2.predict_proba(tmp)[0,1])
print('Difference:', clf2.predict_proba(tmp)[0,1] - clf2.predict_proba(X_test1[idx])[0,1])


# #### Visualizing explanations of toxic comment in imbalance dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi.html')


# In[ ]:


exp.show_in_notebook(text=True)


# After Oversampling 

# In[ ]:


idx = 141
exp = explainer.explain_instance(X_test[idx], c2.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c2.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


print('Original prediction:', clf.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['shit']] = 0
tmp[0,vectorizer.vocabulary_['bastard']] = 0
print('Prediction removing some features:', clf.predict_proba(tmp)[0,1])
print('Difference:', clf.predict_proba(tmp)[0,1] - clf.predict_proba(X_test1[idx])[0,1])


# #### Visualizing explanations for toxic comment after Oversampling the dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi.html')


# In[ ]:


exp.show_in_notebook(text=True)


# After Undersampling

# In[ ]:


idx = 141
exp = explainer.explain_instance(X_test[idx], c3.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c3.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


print('Original prediction:', clf1.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['shit']] = 0
tmp[0,vectorizer.vocabulary_['bastard']] = 0
print('Prediction removing some features:', clf1.predict_proba(tmp)[0,1])
print('Difference:', clf1.predict_proba(tmp)[0,1] - clf1.predict_proba(X_test1[idx])[0,1])


# #### Visualizing explanations for toxic comment after Undersampling the dataset

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi.html')


# In[ ]:


exp.show_in_notebook(text=True)

