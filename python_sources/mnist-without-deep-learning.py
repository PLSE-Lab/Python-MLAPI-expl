#!/usr/bin/env python
# coding: utf-8

# In this kernel I try to find the best classification results without deep learning as I am a machine learning beginner. Of course it will not perform as a Convolutional Neural Network or other fancy deep learning techniques.

# ### Preprocessing

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# Let's have a look at the data:

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
df_train = pd.read_csv('../input/train.csv')
df_train.head()


# In[ ]:


df_test.head()


# Check the ditribution of the labels (1, 2, ..., 9):

# In[ ]:


df_train['label'].hist()
plt.show()


# Create a test set with the same distribution of labels:

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit 

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
for train_index, test_index in split.split(df_train, df_train['label']):
    strat_train_set = df_train.loc[train_index]
    strat_test_set = df_train.loc[test_index]
    
strat_train_set['label'].hist()
strat_test_set['label'].hist()
plt.show()


# Separating the target value array y from the features:

# In[ ]:


y_train = strat_train_set['label'].values
strat_train_set.drop('label', axis=1, inplace=True)
X_train = strat_train_set.values


# In[ ]:


y_test = strat_test_set['label'].values
strat_test_set.drop('label', axis=1, inplace=True)
X_test = strat_test_set.values


# In[ ]:


X_final = df_test.values


# In[ ]:


X_train.shape


# Plot some random digit:

# In[ ]:


some_digit = X_train[9]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
print(y_train[9])
plt.show()


# ### Scale the features:

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

scaler = StandardScaler()
X_final_scaled = scaler.fit_transform(X_final)


# ### Train the classifier

# Let's have a look how some typical classifiers perform out of the box without any parameter tuning (as this takes too long for the kaggle kernel nothing runs):

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def classify(classifier):
    y_pred = cross_val_predict(classifier, X_train, y_train, cv=2, n_jobs=-1, verbose=3)
    return accuracy_score(y_train, y_pred)
  
svc_score = 0.0 #classify(SVC())
knn_score = 0.0 #classify(KNeighborsClassifier())
dt_score = 0.0 #classify(DecisionTreeClassifier())
rf_score = 0.0 #classify(RandomForestClassifier())
ada_score = 0.0 #classify(AdaBoostClassifier())
gau_score = 0.0 #classify(GaussianNB())


# In[ ]:


models = pd.DataFrame({
    'Model': ['SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier','GaussianNB'],
    'Score': [svc_score, knn_score, dt_score, rf_score, ada_score, gau_score]})
models.sort_values(by='Score', ascending=False)


# Grid search the best parameters for KNN. Attention: This will take like forever :-(
# Thats's why I have inclued only the best parameters into the classifier:
# 
# {'n_neighbors': 4, 'weights': 'distance'}

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

search_grid = [
    {'n_neighbors': [4], 'weights': ['distance'], 'n_jobs': [-1]}
]

#grid_search = GridSearchCV(KNeighborsClassifier(), search_grid, cv=3, scoring='accuracy', verbose=3)
#grid_search.fit(X_train_scaled, y_train)

knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
knn_clf.fit(X_train_scaled, y_train)


# In[ ]:


y_knn_pred = knn_clf.predict(X_test_scaled)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_knn_pred)


# In[ ]:


y_knn_pred = knn_clf.predict(X_final_scaled)
y_knn_pred


# ### Write the output file

# In[ ]:


submission = pd.DataFrame({
        "ImageId": list(range(1,len(y_knn_pred)+1)),
        "Label": y_knn_pred
    })
print(submission.head())
submission.to_csv('submission.csv', index=False)

