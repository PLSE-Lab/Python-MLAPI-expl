#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
#from sklearn.datasets import load_iris
# Initializing models


clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = MultinomialNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4], 
                          meta_classifier=lr)

params = {'kneighborsclassifier__n_neighbors': [1, 5],
          'randomforestclassifier__n_estimators': [10, 50],
          'meta-logisticregression__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    refit=True)
cv_keys = ('mean_test_score', 'std_test_score', 'params')


# In[3]:


import os
os.listdir('../input')
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')


# In[5]:


from sklearn import preprocessing
data_cleaner =[train,test]
label = preprocessing.LabelEncoder()
for data in data_cleaner:
    #data['teacher_prefix_code']=label.fit_transform(data['teacher_prefix'])
    data['project_subject_categories_code']=label.fit_transform(data['project_subject_categories'])
    data['project_subject_subcategories_code']=label.fit_transform(data['project_subject_subcategories'])
    data['project_grade_category_code']=label.fit_transform(data['project_grade_category'])


# In[6]:


data_cleaner =[train,test]

test['teacher_prefix'].fillna(test['teacher_prefix'].mode()[0], inplace = True)


# In[7]:


columns = ['project_grade_category_code','project_subject_subcategories_code','project_subject_categories_code','teacher_number_of_previously_posted_projects']
y_true = train['project_is_approved']
x = train[columns]


# In[8]:


grid.fit(x,y_true)


# In[9]:


grid.best_score_


# In[10]:


for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)


# In[11]:


x_test = test[columns]
y_pred1 = grid.predict_proba(x_test)[:,1]


# In[12]:


y_pred = grid.predict(x_test)


# In[13]:


sub = sample
sub1 = sample
sub.project_is_approved=y_pred
sub1.project_is_approved=y_pred1


# In[14]:


sub1.to_csv('Submission_Proba',index=False)
sub.to_csv('Submission_prediction',index=False)

