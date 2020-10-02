#!/usr/bin/env python
# coding: utf-8

# SO this is a basic kernal describing how to use the LIME , further i have used this concept to build two kernals, in very detailed manner.
# 
# 1. In this kernal i have used Logistic Regression Classifier to predict the class of comment (toxic or non-toxic)[kernal](https://www.kaggle.com/bavalpreet26/explainable-ai-lime)
# 
# 2. In this 2nd kernal i have used Support Vector Classifier with same pipeline to predict the class of comment (toxic or non-toxic)[kernal2](https://www.kaggle.com/bavalpreet26/explainable-ai-lime-svc)

# # Why Explainable AI is so Important?
# 
# Explainable AI (XAI) refers to methods and techniques in the application of artificial intelligence technology (AI) such that the results of the solution can be understood by human beings. It contrasts with the concept of the "black box" in machine learning where even their designers cannot explain why the AI arrived at a specific decision.

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTz0dmBOdIz5VNjYqkK4kwK4467a0AR3Iro8YxzHF8IxjSyvsFY&usqp=CAU)

# ![](https://miro.medium.com/max/4552/1*XN9NNL_Q3EcctTELM3CoNA.png)

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


from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics


# In[ ]:


from sklearn.datasets import fetch_20newsgroups
categories = ['rec.autos', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['autos', 'med']


# In[ ]:


newsgroups_train.data[9:10]


# In[ ]:


type(newsgroups_train.data)


# In[ ]:


type(newsgroups_train.target)


# In[ ]:


newsgroups_train.target[:10]


# we will now use tf/idf vectorizer

# In[ ]:


vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)


# In[ ]:


type(vectorizer)


# Now, let's say we want to use random forests for classification. It's usually hard to understand what random forests are doing, especially with many trees.

# In[ ]:


rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)


# In[ ]:


pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')


# In[ ]:


pred[:10]


# ### Explaining predictions using lime

# Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on vectorized representation of texts. For this purpose, we use sklearn's pipeline, and implements predict_proba on raw_text lists.

# In[ ]:


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)


# In[ ]:


type(newsgroups_test.data)


# In[ ]:


newsgroups_test.data[0]


# In[ ]:


newsgroups_test.target[0]


# In[ ]:


type(newsgroups_test.target)


# In[ ]:


print(c.predict_proba([newsgroups_test.data[0]]))


# so we get higher prob for med class and low for autos

# Now we create an explainer object. We pass the class_names a an argument for prettier display.

# In[ ]:


from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# We then generate an explanation with at most 6 features for an arbitrary document in the test set.

# In[ ]:


idx = 0
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(med) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])


# The classifier got this example right (it predicted med).
# The explanation is presented below as a list of weighted features.

# In[ ]:


exp.as_list()


# These weighted features are a linear model, which approximates the behaviour of the random forest classifier in the vicinity of the test example. Roughly, if we remove 'doctor' and 'is' from the document , the prediction should move towards the opposite class (autos) by about 0.07 (the sum of the weights for both features). Let's see if this is the case.

# In[ ]:


print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['doctor']] = 0
tmp[0,vectorizer.vocabulary_['is']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])


# Pretty close!
# The words that explain the model around this document seem somewhat related like 'doctor' and 'medical'.And from remaining 'is' and 'of' are kinda stopwords whereas 'effects' or 'information' i think not much to do with or somehow very less related to medical
# 

# ### Visualizing explanations

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi.html')


# In[ ]:


exp.show_in_notebook(text=True)


# In[ ]:


print(c.predict_proba([newsgroups_test.data[1]]))
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# In[ ]:


idx = 1
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(med) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['of']] = 0
tmp[0,vectorizer.vocabulary_['subject']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=False)


# In[ ]:


exp.save_to_file('/tmp/oi1.html')


# In[ ]:


exp.show_in_notebook(text=True)


# SO this is a basic kernal describing how to use the LIME , further i have used this concept to build two kernals, in very detailed manner.
# 
# 1. In this kernal i have used Logistic Regression Classifier to predict the class of comment (toxic or non-toxic)[kernal](https://www.kaggle.com/bavalpreet26/explainable-ai-lime)
# 
# 2. In this 2nd kernal i have used Support Vector Classifier with same pipeline to predict the class of comment (toxic or non-toxic)[kernal2](https://www.kaggle.com/bavalpreet26/explainable-ai-lime-svc)
