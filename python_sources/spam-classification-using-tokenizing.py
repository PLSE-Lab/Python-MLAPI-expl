#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/SPAM text message 20170820 - Data.csv")


# In[ ]:


data.head()


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['Category'] = le.fit_transform(data['Category'])


# In[ ]:


X = data.iloc[:,1]
y = data.iloc[:,0]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 40)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'TfidfVectorizer')


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,3),dtype=np.float32)


# In[ ]:


X_train_vect = vect_word.fit_transform(X_train)
X_test_vect = vect_word.fit_transform(X_test)


# In[ ]:


print(X_train_vect.toarray())


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
rf.fit(X_train_vect, y_train)


# In[ ]:


y_pred = rf.predict(X_test_vect)


# In[ ]:


from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
print("\nConfusion Matrix\n",confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vect_word, rf)


# In[ ]:


print(c.predict_proba(X_test[0:1]))


# In[ ]:


from lime.lime_text import LimeTextExplainer
class_names = ['ham','spam']
explainer = LimeTextExplainer(class_names=class_names)


# In[ ]:


idx = 0
exp = explainer.explain_instance(X_test[0], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability =', c.predict_proba(X_test[0:1])[0,1])
print('True class: %s' % class_names[y_test[0]])


# In[ ]:


exp.as_list()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[ ]:


exp.show_in_notebook(text=True)

