#!/usr/bin/env python
# coding: utf-8

# 

# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# 

# In[ ]:


#Map salary to 0,1,2
df.salary = df.salary.map({'low':0,'medium':1,'high':2})


# In[ ]:


#Generate X and y
X = df.drop(['left','sales'],1)
y = df['left']


# 

# In[ ]:


import numpy as np
from sklearn import preprocessing,cross_validation,neighbors,svm


# In[ ]:


#splitting the train and test sets
X_train, X_test, y_train,y_test= cross_validation.train_test_split(X,y,test_size=0.2)


# 

# In[ ]:


clf = svm.SVC()
clf.fit(X_train,y_train)


# In[ ]:


pd.DataFrame(X_train,y_train).head()


# In[ ]:


accuracy = clf.score(X_test,y_test)


# In[ ]:


print(accuracy)


# In[ ]:


pd.DataFrame(clf.predict(X_test),y_test,columns=['ytest']).head()


# 

# In[ ]:


from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[ ]:


sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()


# In[ ]:


# Predict Test Set
favorite_clf = RandomForestClassifier()
favorite_clf.fit(X_train, y_train)
submission = pd.DataFrame(favorite_clf.predict(X_test))


# In[ ]:


# Export Submission
# submission.to_csv('../Output/submission.csv', index = False)
submission.tail()


# 
