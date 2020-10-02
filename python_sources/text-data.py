#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
random_seed = 5

# Any results you write to the current directory are saved as output.


# In[ ]:


reuters = pd.read_csv("../input/reuters-newswire-2017-edited.csv")


# In[ ]:


reuters = reuters[reuters.category!=7]


# In[ ]:


reuters.head()


# In[ ]:


vec = TfidfVectorizer(ngram_range=(1,3))


# In[ ]:


vec.fit(reuters.headline_text.values)


# In[ ]:


feature =  vec.transform(reuters.headline_text.values).toarray()


# In[ ]:


reuters.category.value_counts()


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(feature, reuters.category,random_state=random_seed,stratify=reuters.category)


# In[ ]:


Y_test.shape


# In[ ]:


len(vec.get_feature_names())


# In[ ]:


Rm = RandomForestClassifier(n_estimators=200)


# In[ ]:


Rm.fit(X_train,Y_train)


# In[ ]:


Rm.score(X_test,Y_test)


# In[ ]:


Rm.score(X_train,Y_train)


# In[ ]:


Y_pred = Rm.predict(X_test)


# In[ ]:


accuracy_score(Y_test,Y_pred)


# In[ ]:


Rm.feature_importances_


# In[ ]:


featureimportance = pd.DataFrame({"feature":vec.get_feature_names(),"importance":Rm.feature_importances_})


# In[ ]:


featureimportance.head()


# In[ ]:


pltdata = featureimportance.sort_values("importance",ascending=False)[featureimportance.importance>0.001]


# In[ ]:


sns.set(font_scale=1.2)
plt.figure(figsize = (16,6))
sns.barplot(data = pltdata,x='feature',y='importance')
plt.xticks(rotation=90);


# In[ ]:





# In[ ]:




