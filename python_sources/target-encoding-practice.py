#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/mushrooms.csv')


# In[ ]:


df.head(10)


# In[ ]:


df['class'].value_counts()


# In[ ]:


for c in df.columns:
    if c != 'class':
        df.groupby([c,'class']).count().unstack().iloc[:,0:2].plot(kind='barh')


# In[ ]:


df['label'] = df['class'].replace('e',0).replace('p',1)


# In[ ]:


#target encoding
for c in df.columns:
    if c not in ('label','class'):
        vals = df[['label',c]].groupby(c).mean()
        d = vals.to_dict()['label']
        df[c + '_numeric'] = df[c].replace(d)
    


# In[ ]:


df.head(10)


# In[ ]:


df.columns[24]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,classification_report

y = df['label']
X = df.iloc[:,24:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

param_grid = [{'n_estimators':[10,20,50,100]}]
RFC = RandomForestClassifier()
cv = GridSearchCV(RFC,param_grid,verbose=0,cv=5)


# In[ ]:


cv.fit(X_train,y_train)


# In[ ]:


confusion_matrix(y_test,cv.predict(X_test))


# In[ ]:


pd.Series(cv.best_estimator_.feature_importances_,index=df.columns[24:]).sort_values().plot(kind='barh')


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,classification_report

y = df['label']
X = df.iloc[:,24:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

param_grid = [{'n_estimators':[10,20,50,100]}]
GBC = GradientBoostingClassifier()
cv = GridSearchCV(GBC,param_grid,verbose=0,cv=5)


# In[ ]:


cv.fit(X_train,y_train)


# In[ ]:


confusion_matrix(y_test,cv.predict(X_test))


# In[ ]:


pd.Series(cv.best_estimator_.feature_importances_,index=df.columns[24:]).sort_values().plot(kind='barh')


# In[ ]:




