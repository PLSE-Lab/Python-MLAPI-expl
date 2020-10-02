#!/usr/bin/env python
# coding: utf-8

# # Load the iris dataset, the ensemble library for sklearn and tree for creating base estimator from sklearn

# In[ ]:


from sklearn.datasets import load_iris
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree


# # Load Iris dataset and split the train and test data

# In[ ]:


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Create a base estimator. A stump in this case

# In[ ]:


stump = tree.DecisionTreeClassifier(max_depth=1)


# # Create an AdaBoost classifier with the weak learner - stump in this case

# In[ ]:


clf = ensemble.AdaBoostClassifier(base_estimator = stump, algorithm="SAMME", n_estimators=6, random_state=0)
clf = clf.fit(X_train, y_train)


# # Predict the test data

# In[ ]:


y_test_predicted = clf.predict(X_test)


# # Print confusion matrix and accuracy

# In[ ]:


print(confusion_matrix(y_test, y_test_predicted))
accuracy_score(y_test, y_test_predicted)


# In[ ]:




