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


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


# In[ ]:


f1=pd.read_json('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json',lines=True)
f2=pd.read_json('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json',lines=True)


# In[ ]:


df=pd.concat([f1,f2],axis=0)
df.head()


# In[ ]:


df.info()


# In[ ]:



print('0 : ',df[df['is_sarcastic']==0].count())
print('1 : ',df[df['is_sarcastic']==1].count())


# #### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',RandomForestClassifier())])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print(accuracy_score(y_test,y_pred))


# #### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',LogisticRegression())])
model=pipe.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #### Support Vector Classifier

# In[ ]:


from sklearn.svm import SVC,LinearSVC


# In[ ]:


X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


model=LinearSVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',LinearSVC())])
pipe.fit(X_train_,y_train_)
y_pred=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred))


# In[ ]:


grid_param={'C':[0.1,1,10],
           'gamma':[1,0.1,0.001]}


# In[ ]:


grid=GridSearchCV(SVC(),param_grid=grid_param,refit=True,verbose=3)
grid.fit(X_train,y_train)


# In[ ]:


model=SVC(C=1,gamma=1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


from sklearn.linear_model import SGDClassifier


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',SGDClassifier())])


# In[ ]:


model=pipe.fit(X_train_,y_train_)
y_pred_=model.predict(X_test_)


# In[ ]:


print(accuracy_score(y_test_,y_pred_))


# #### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


model=MultinomialNB()
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',MultinomialNB())])
pipe.fit(X_train_,y_train_)


# In[ ]:


y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))


# In[ ]:


from sklearn.naive_bayes import BernoulliNB


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',BernoulliNB())])
pipe.fit(X_train_,y_train_)


# In[ ]:


y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))


# In[ ]:


para={'alpha':[1,0.5,0.1,0.01,0]}


# In[ ]:


model=GridSearchCV(estimator=BernoulliNB(),param_grid=para,n_jobs=-1,cv=3,verbose=3)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# #### Gradient Boost Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(TfidfVectorizer().fit_transform(df['headline']),
                                              df['is_sarcastic'],
                                              test_size=0.2,
                                               random_state=101)
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],
                                                   df['is_sarcastic'],
                                                   test_size=0.2,
                                                   random_state=101)


# In[ ]:


model=GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
             ('tdidf',TfidfTransformer()),
             ('model',GradientBoostingClassifier())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))


# #### XGBoost Classifier

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',XGBClassifier())])
pipe.fit(X_train_,y_train_)


# In[ ]:


y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))


# #### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(TfidfVectorizer().fit_transform(df['headline']),
                                              df['is_sarcastic'],
                                              test_size=0.2,
                                              random_state=101)
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],df['is_sarcastic'],
                                                  test_size=0.2,
                                                  random_state=101)


# In[ ]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


params = {'max_leaf_nodes': list(range(2, 100,10)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)


# In[ ]:


grid_search_cv.best_params_


# In[ ]:


model=grid_search_cv.best_estimator_


# In[ ]:


model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',DecisionTreeClassifier())])
pipe.fit(X_train_,y_train_)


# In[ ]:


y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))


# #### KNN Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(TfidfVectorizer().fit_transform(df['headline']),
                                              df['is_sarcastic'],
                                              test_size=0.2,
                                              random_state=101)
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],
                                                  df['is_sarcastic'],
                                                  test_size=0.2,random_state=101)


# In[ ]:


model=KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


grid=GridSearchCV(estimator=KNeighborsClassifier(),param_grid={'n_neighbors':[1,2,10,20,50]},verbose=3,n_jobs=1)
grid.fit(X_train,y_train)


# In[ ]:


model=grid.best_estimator_
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:


pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',KNeighborsClassifier())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))


# In[ ]:




