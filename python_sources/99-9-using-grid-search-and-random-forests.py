#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/HR_comma_sep.csv')
data = data.rename(columns={'sales': 'dept'})


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder

class preproc():
    
    def __init__(self, data, cols):
        self.data = data
        
    def transform(self, dummies=False):
        if dummies:
            print('getting dummies for cat. variables...')
            self.data = pd.get_dummies(self.data, columns=cols)
            return self.data
        else:
            for col in cols:
                print('label encoding...')
                le = LabelEncoder()
                le.fit(self.data[col])
                self.data[col] = le.transform(self.data[col]) 
                print(le.classes_)
            return self.data


# In[ ]:


cols = ['dept', 'salary']
pp = preproc(data, cols)
data = pp.transform(dummies=False)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data.drop('left', axis=1), data['left'], test_size=0.3)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


# In[ ]:


from sklearn.feature_selection import RFECV


# In[ ]:


rf = RandomForestClassifierWithCoef()
rfecv = RFECV(estimator=rf, step=1, cv=3, scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)


# In[ ]:


X_train.columns.values


# In[ ]:


rfecv.ranking_


# In[ ]:


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:


#X_train = X_train.loc[:,rfecv.support_]


# In[ ]:


X_train.columns.values


# In[ ]:


pipe = Pipeline([
    ('clf', RandomForestClassifier(random_state=0))
])


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nparam_grid = {\n        'clf__n_estimators': [3, 10, 30, 100],\n        'clf__criterion': ['gini', 'entropy'],\n        'clf__class_weight': [None, 'balanced', 'balanced_subsample']\n        }\n\ngrid = GridSearchCV(pipe, cv=3, param_grid=param_grid, scoring='accuracy')\ngrid.fit(X_train, y_train)")


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


grid.cv_results_['mean_test_score']


# In[ ]:




